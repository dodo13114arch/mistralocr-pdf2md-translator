#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF Mistral OCR 匯出工具

本程式可將 PDF 文件自動化轉換為 Markdown 格式，包含以下流程：

1. 使用 Mistral OCR 模型辨識 PDF 內文與圖片
2. 將辨識結果組成含圖片的 Markdown 檔
3. 使用 Gemini 模型將英文內容翻譯為台灣繁體中文
4. 匯出 Markdown 檔（原文版 + 翻譯版）與對應圖片

新增功能：
- 處理過程中的檢查點，可以保存中間結果
- Gradio 介面，方便調整參數和選擇輸出格式
"""

# Standard libraries
import os
import json
import base64
import time
from pathlib import Path
import pickle
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# Third-party libraries
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr

# Mistral AI
from mistralai import Mistral
from mistralai.models import OCRResponse, ImageURLChunk, DocumentURLChunk, TextChunk

# Google Gemini
from google import genai
from google.genai import types

# OpenAI
# Import the library (add 'openai' to requirements.txt)
try:
    from openai import OpenAI
except ImportError:
    print("⚠️ OpenAI library not found. Please install it: pip install openai")
    OpenAI = None # Set to None if import fails

# ===== Pydantic Models =====

class StructuredOCR(BaseModel):
    file_name: str
    topics: list[str]
    languages: str
    ocr_contents: dict

# ===== Utility Functions =====

def retry_with_backoff(func, retries=5, base_delay=1.5):
    """Retry a function with exponential backoff."""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e):
                wait_time = base_delay * (2 ** attempt)
                print(f"⚠️ API rate limit hit. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("❌ Failed after multiple retries.")

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders in markdown with base64-encoded images."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """Combine OCR text and images into a single markdown document."""
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {img.id: img.image_base64 for img in page.images}
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

def insert_ocr_below_images(markdown_str, ocr_img_map, page_idx):
    """Insert OCR results below images in markdown."""
    for img_id, ocr_text in ocr_img_map.get(page_idx, {}).items():
        markdown_str = markdown_str.replace(
            f"![{img_id}]({img_id})",
            f"![{img_id}]({img_id})\n\n> 📄 Image OCR Result：\n\n```json\n{ocr_text}\n```"
        )
    return markdown_str

def save_images_and_replace_links(markdown_str, images_dict, page_idx, image_folder="images"):
    """Save base64 images to files and update markdown links."""
    os.makedirs(image_folder, exist_ok=True)
    image_id_to_path = {}

    for i, (img_id, base64_str) in enumerate(images_dict.items()):
        img_bytes = base64.b64decode(base64_str.split(",")[-1])
        # 使用相對路徑，僅保留資料夾名稱和檔案名稱
        img_path = f"{os.path.basename(image_folder)}/page_{page_idx+1}_img_{i+1}.png"
        
        # 實際儲存的完整路徑
        full_img_path = os.path.join(image_folder, f"page_{page_idx+1}_img_{i+1}.png")
        with open(full_img_path, "wb") as f:
            f.write(img_bytes)
        image_id_to_path[img_id] = img_path

    for img_id, img_path in image_id_to_path.items():
        markdown_str = markdown_str.replace(
            f"![{img_id}]({img_id})", f"![{img_id}]({img_path})"
        )

    return markdown_str

# ===== Translation Functions =====

# Default translation system prompt
DEFAULT_TRANSLATION_SYSTEM_INSTRUCTION = """
你是一位專業的技術文件翻譯者。請將我提供的英文 Markdown 內容翻譯成**台灣繁體中文**。

**核心要求：**
1.  **翻譯所有英文文字：** 你的主要工作是翻譯內容中的英文敘述性文字（段落、列表、表格等）。
2.  **保持結構與程式碼不變：**
    * **不要**更改任何 Markdown 標記（如 `#`, `*`, `-`, `[]()`, `![]()`, ``` ```, ` `` `, `---`）。
    * **不要**翻譯或修改程式碼區塊 (``` ... ```) 和行內程式碼 (`code`) 裡的任何內容。
    * 若有 JSON，**不要**更改鍵（key），僅翻譯字串值（value）。
3.  **處理專有名詞：** 對於普遍接受的英文技術術語、縮寫或專有名詞（例如 API, SDK, CPU, Google, Python 等），傾向於**保留英文原文**。但請確保翻譯了其他所有非術語的常規英文文字。
4.  **直接輸出結果：** 請直接回傳翻譯後的完整 Markdown 文件，不要添加任何額外說明。
"""

# Updated signature to accept openai_client
def translate_markdown_pages(pages, gemini_client, openai_client, model="gemini-2.0-flash", system_instruction=None):
    """Translate markdown pages using the selected API (Gemini or OpenAI). Yields progress strings and translated page content."""
    if system_instruction is None:
        system_instruction = DEFAULT_TRANSLATION_SYSTEM_INSTRUCTION

    # No longer collecting in a list here, will yield pages directly
    total_pages = len(pages) # Get total pages for progress

    for idx, page in enumerate(pages):
        progress_message = f"🔁 正在翻譯第 {idx+1} / {total_pages} 頁..."
        print(progress_message) # Print to console
        yield progress_message # Yield progress string for Gradio log

        try:
            if model.startswith("gpt-"):
                # --- OpenAI Translation Logic ---
                if not openai_client:
                    error_msg = f"⚠️ OpenAI client not initialized for translation model {model}. Skipping page {idx+1}."
                    print(error_msg)
                    yield error_msg
                    yield f"--- ERROR: OpenAI Client Error for Page {idx+1} ---\n\n{page}"
                    continue # Skip to next page

                print(f"    - Translating using OpenAI model: {model}")
                try:
                    # Construct messages for OpenAI translation
                    # Use the provided system_instruction as the system message
                    messages = [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": page} 
                    ]
                    
                    response = openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.1 # Lower temperature for more deterministic translation
                    )
                    translated_md = response.choices[0].message.content.strip()
                except Exception as openai_e:
                    error_msg = f"⚠️ OpenAI 翻譯第 {idx+1} / {total_pages} 頁失敗：{openai_e}"
                    print(error_msg)
                    yield error_msg # Yield error string to Gradio log
                    yield f"--- ERROR: OpenAI Translation Failed for Page {idx+1} ---\n\n{page}"
                    continue # Skip to next page

            elif model.startswith("gemini"):
                # --- Gemini Translation Logic ---
                print(f"    - Translating using Gemini model: {model}")
                response = gemini_client.models.generate_content(
                    model=model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction
                    ),
                    contents=page
                )
                translated_md = response.text.strip()
            
            else:
                # --- Unsupported Model ---
                error_msg = f"⚠️ Unsupported translation model: {model}. Skipping page {idx+1}."
                print(error_msg)
                yield error_msg
                yield f"--- ERROR: Unsupported Translation Model for Page {idx+1} ---\n\n{page}"
                continue # Skip to next page

            # --- Yield successful translation ---
            # translated_pages.append(translated_md) # Removed duplicate append

            yield translated_md # Yield the actual translated page content

        except Exception as e:
            error_msg = f"⚠️ 翻譯第 {idx+1} / {total_pages} 頁失敗：{e}"
            print(error_msg)
            yield error_msg # Yield error string to Gradio log
            # Yield error marker instead of translated content
            yield f"--- ERROR: Translation Failed for Page {idx+1} ---\n\n{page}"

    final_message = f"✅ 翻譯完成 {total_pages} 頁。"
    yield final_message # Yield final translation status string
    print(final_message) # Print final translation status
    # No return needed for a generator yielding results

# ===== PDF Processing Functions =====

def process_pdf_with_mistral_ocr(pdf_path, client, model="mistral-ocr-latest"):
    """Process PDF with Mistral OCR."""
    pdf_file = Path(pdf_path)
    
    # Upload to mistral
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_file.stem,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr"
    )
    
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    
    # OCR analyze PDF
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model=model,
        include_image_base64=True
    )
    
    return pdf_response

# Updated function signature to include structure_text_only
def process_images_with_ocr(pdf_response, mistral_client, gemini_client, openai_client, structure_model="pixtral-12b-latest", structure_text_only=False):
    """Process images from PDF pages with OCR and structure using the specified model."""
    image_ocr_results = {}
    
    for page_idx, page in enumerate(pdf_response.pages):
        for i, img in enumerate(page.images):
            base64_data_url = img.image_base64
            
            # Extract raw base64 data for Gemini
            try:
                # Handle potential variations in data URL prefix
                if ',' in base64_data_url:
                    base64_content = base64_data_url.split(',', 1)[1]
                else:
                    # Assume it's just the base64 content if no comma prefix
                    base64_content = base64_data_url 
                # Decode and re-encode to ensure it's valid base64 bytes for Gemini
                image_bytes = base64.b64decode(base64_content)
            except Exception as e:
                print(f"⚠️ Error decoding base64 for page {page_idx+1}, image {i+1}: {e}. Skipping image.")
                continue # Skip this image if base64 is invalid

            def run_ocr_and_parse():
                # Step 1: Basic OCR (always use Mistral OCR for initial text extraction)
                print(f"  - Performing basic OCR on page {page_idx+1}, image {i+1}...")
                image_response = mistral_client.ocr.process(
                    document=ImageURLChunk(image_url=base64_data_url),
                    model="mistral-ocr-latest" # Use the dedicated OCR model here
                )
                image_ocr_markdown = image_response.pages[0].markdown
                print("  - Basic OCR text extracted.")

                # Step 2: Structure the OCR markdown using the selected model
                print(f"  - Structuring OCR using: {structure_model}")
                if structure_model == "pixtral-12b-latest":
                    print("    - Using Mistral Pixtral...")
                    print("    - Sending request to Pixtral API...") # Added print statement
                    structured = mistral_client.chat.parse(
                        model=structure_model, # Use the selected structure_model
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    ImageURLChunk(image_url=base64_data_url),
                                    TextChunk(text=(
                                        f"This is the image's OCR in markdown:\n{image_ocr_markdown}\n. "
                                        "Convert this into a structured JSON response with the OCR contents in a sensible dictionary."
                                    ))
                                ]
                            }
                        ],
                        response_format=StructuredOCR, # Use Pydantic model for expected structure
                        temperature=0
                    )
                    structured_data = structured.choices[0].message.parsed
                    pretty_text = json.dumps(structured_data.ocr_contents, indent=2, ensure_ascii=False)

                elif structure_model.startswith("gemini"): # Handle gemini-flash-2.0 etc.
                    print(f"    - Using Google Gemini ({structure_model})...")
                    # Define the base prompt text
                    base_prompt_text = f"""
You are an expert OCR structuring assistant. Your goal is to extract and structure the relevant content into a JSON object based on the provided information.

**Initial OCR Markdown:**
```markdown
{image_ocr_markdown}
```

**Task:**
Generate a JSON object containing the structured OCR content found in the image. Focus on extracting meaningful information and organizing it logically within the JSON. The JSON should represent the `ocr_contents` field.

**Output Format:**
Return ONLY the JSON object, without any surrounding text or markdown formatting. Example:
```json
{{
  "title": "Example Title",
  "sections": [
    {{"header": "Section 1", "content": "Details..."}},
    {{"header": "Section 2", "content": "More details..."}}
  ],
  "key_value_pairs": {{
    "key1": "value1",
    "key2": "value2"
  }}
}}
```
(Adapt the structure based on the image content.)
"""
                    # Prepare API call based on structure_text_only flag
                    gemini_contents = []
                    if structure_text_only:
                        print("    - Mode: Text-only structuring")
                        # Modify prompt slightly for text-only
                        gemini_prompt = base_prompt_text.replace(
                            "Analyze the provided image and the initial OCR text", 
                            "Analyze the initial OCR text"
                        ).replace(
                            "content from the image",
                            "content from the text"
                        )
                        gemini_contents.append(gemini_prompt)
                    else:
                        print("    - Mode: Image + Text structuring")
                        gemini_prompt = base_prompt_text # Use original prompt
                        # Prepare image part for Gemini using types.Part.from_bytes
                        # Assuming PNG, might need dynamic type detection in the future
                        # Pass the decoded image_bytes, not the base64_content string
                        try: # Corrected indentation
                            image_part = types.Part.from_bytes(
                                mime_type="image/png", 
                                data=image_bytes 
                            )
                            gemini_contents = [gemini_prompt, image_part] # Text prompt first, then image Part
                        except Exception as e:
                             print(f"    - ⚠️ Error creating Gemini image Part: {e}. Skipping image structuring.")
                             # Fallback or re-raise depending on desired behavior
                             pretty_text = json.dumps({"error": "Failed to create Gemini image Part", "details": str(e)}, indent=2, ensure_ascii=False)
                             return pretty_text # Exit run_ocr_and_parse for this image

                    # Call Gemini API - Corrected to use gemini_client.models.generate_content
                    print(f"    - Sending request to Gemini API ({structure_model})...") # Added print statement
                    
                    try:
                        response = gemini_client.models.generate_content(
                            model=structure_model, 
                            contents=gemini_contents # Pass the constructed list
                        )
                    except Exception as api_e:
                         print(f"    - ⚠️ Error calling Gemini API: {api_e}")
                         # Fallback or re-raise
                         pretty_text = json.dumps({"error": "Failed to call Gemini API", "details": str(api_e)}, indent=2, ensure_ascii=False)
                         return pretty_text # Exit run_ocr_and_parse for this image
                    
                    # Extract and clean the JSON response
                    raw_json_text = response.text.strip()
                    # Remove potential markdown code fences
                    if raw_json_text.startswith("```json"):
                        raw_json_text = raw_json_text[7:]
                    if raw_json_text.endswith("```"):
                        raw_json_text = raw_json_text[:-3]
                    raw_json_text = raw_json_text.strip()

                    # Validate and format the JSON
                    try:
                        parsed_json = json.loads(raw_json_text)
                        pretty_text = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError as json_e:
                        print(f"    - ⚠️ Gemini response was not valid JSON: {json_e}")
                        print(f"    - Raw response: {raw_json_text}")
                        # Fallback: return the raw text wrapped in a basic JSON structure
                        pretty_text = json.dumps({"error": "Failed to parse Gemini JSON response", "raw_output": raw_json_text}, indent=2, ensure_ascii=False)

                elif structure_model.startswith("gpt-"):
                    print(f"    - Using OpenAI model: {structure_model}...")
                    if not openai_client:
                        print("    - ⚠️ OpenAI client not initialized. Skipping.")
                        return json.dumps({"error": "OpenAI client not initialized. Check API key and library installation."}, indent=2, ensure_ascii=False)

                    # Define the base prompt text for OpenAI
                    openai_base_prompt = f"""
You are an expert OCR structuring assistant. Your goal is to extract and structure the relevant content into a JSON object based on the provided information.

**Initial OCR Markdown:**
```markdown
{image_ocr_markdown}
```

**Task:**
Generate a JSON object containing the structured OCR content found in the image. Focus on extracting meaningful information and organizing it logically within the JSON. The JSON should represent the `ocr_contents` field.

**Output Format:**
Return ONLY the JSON object, without any surrounding text or markdown formatting. Example:
```json
{{
  "title": "Example Title",
  "sections": [
    {{"header": "Section 1", "content": "Details..."}},
    {{"header": "Section 2", "content": "More details..."}}
  ],
  "key_value_pairs": {{
    "key1": "value1",
    "key2": "value2"
  }}
}}
```
(Adapt the structure based on the image content. Ensure the output is valid JSON.)
"""
                    # Prepare payload for OpenAI vision based on structure_text_only
                    openai_content_list = []
                    if structure_text_only:
                        print("    - Mode: Text-only structuring")
                        # Modify prompt slightly for text-only
                        openai_prompt = openai_base_prompt.replace(
                            "Analyze the provided image and the initial OCR text", 
                            "Analyze the initial OCR text"
                        ).replace(
                            "content from the image",
                            "content from the text"
                        )
                        openai_content_list.append({"type": "text", "text": openai_prompt})
                    else:
                        print("    - Mode: Image + Text structuring")
                        openai_prompt = openai_base_prompt # Use original prompt
                        # Use the base64_content string directly for the data URL
                        # Assuming PNG, might need dynamic type detection
                        image_data_url = f"data:image/png;base64,{base64_content}" # Corrected indentation
                        openai_content_list.append({"type": "text", "text": openai_prompt})
                        openai_content_list.append({
                            "type": "image_url",
                            "image_url": {"url": image_data_url, "detail": "auto"}, 
                        })

                    print(f"    - Sending request to OpenAI API ({structure_model})...")
                    try:
                        response = openai_client.chat.completions.create(
                            model=structure_model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": openai_content_list, # Pass the constructed list
                                }
                            ],
                            # Optionally add max_tokens if needed, but rely on prompt for JSON structure
                            # max_tokens=1000, 
                            temperature=0.1 # Lower temperature for deterministic JSON
                        )
                        
                        raw_json_text = response.choices[0].message.content.strip()
                        # Clean potential markdown fences
                        if raw_json_text.startswith("```json"):
                            raw_json_text = raw_json_text[7:]
                        if raw_json_text.endswith("```"):
                            raw_json_text = raw_json_text[:-3]
                        raw_json_text = raw_json_text.strip()

                        # Validate and format JSON
                        try:
                            parsed_json = json.loads(raw_json_text)
                            pretty_text = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                        except json.JSONDecodeError as json_e:
                            print(f"    - ⚠️ OpenAI response was not valid JSON: {json_e}")
                            print(f"    - Raw response: {raw_json_text}")
                            pretty_text = json.dumps({"error": "Failed to parse OpenAI JSON response", "raw_output": raw_json_text}, indent=2, ensure_ascii=False)

                    except Exception as api_e:
                        print(f"    - ⚠️ Error calling OpenAI API: {api_e}")
                        pretty_text = json.dumps({"error": "Failed to call OpenAI API", "details": str(api_e)}, indent=2, ensure_ascii=False)
                
                else: # Final attempt to correct indentation for the final else
                    print(f"    - ⚠️ Unsupported structure model: {structure_model}. Skipping structuring.")
                    # Fallback: return the basic OCR markdown wrapped in JSON
                    pretty_text = json.dumps({"unstructured_ocr": image_ocr_markdown}, indent=2, ensure_ascii=False)

                return pretty_text
            
            try:
                # Pass the actual structure model name to the inner function if needed,
                # or rely on the outer scope variable 'structure_model' as done here.
                result = retry_with_backoff(run_ocr_and_parse, retries=4)
                image_ocr_results[(page_idx, img.id)] = result
            except Exception as e:
                print(f"❌ Failed at page {page_idx+1}, image {i+1}: {e}")
    
    # Reorganize results by page
    ocr_by_page = {}
    for (page_idx, img_id), ocr_text in image_ocr_results.items():
            ocr_by_page.setdefault(page_idx, {})[img_id] = ocr_text
            print(f"  - Successfully processed page {page_idx+1}, image {i+1} with {structure_model}.")
    
    return ocr_by_page

# ===== Checkpoint Functions =====

def save_checkpoint(data, filename, console_output=None):
    """Save data to a checkpoint file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    message = f"✅ 已儲存檢查點：{filename}"
    print(message) # Corrected indentation
    # Removed console_output append
    return message # Return message

def load_checkpoint(filename, console_output=None):
    """Load data from a checkpoint file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        message = f"✅ 已載入檢查點：{filename}"
        print(message)
        # Removed console_output append
        return data, message # Return message
    return None, None # Return None message

# ===== Main Processing Function =====

# Updated function signature to include structure_text_only
def process_pdf_to_markdown(
    pdf_path, 
    mistral_client, 
    gemini_client,
    openai_client, 
    ocr_model="mistral-ocr-latest",
    structure_model="pixtral-12b-latest",
    structure_text_only=False, # Added structure_text_only
    translation_model="gemini-2.0-flash",
    translation_system_prompt=None,
    process_images=True,
    output_formats_selected=None, # New parameter for selected formats
    output_dir=None,
    checkpoint_dir=None,
    use_existing_checkpoints=True
):
    """Main function to process PDF to markdown with translation. Yields log messages."""
    if output_formats_selected is None:
        output_formats_selected = ["中文翻譯", "英文原文"] # Default if not provided

    pdf_file = Path(pdf_path)
    filename_stem = pdf_file.stem
    # Sanitize the filename stem here as well
    sanitized_stem = filename_stem.replace(" ", "_")
    print(f"--- 開始處理檔案: {pdf_file.name} (Sanitized Stem: {sanitized_stem}) ---") # Console print

    # Output and checkpoint directories are now expected to be set by the caller (Gradio function)
    # os.makedirs(output_dir, exist_ok=True) # Ensure caller created it
    # os.makedirs(checkpoint_dir, exist_ok=True) # Ensure caller created it

    # Checkpoint files - Use sanitized_stem
    pdf_ocr_checkpoint = os.path.join(checkpoint_dir, f"{sanitized_stem}_pdf_ocr.pkl")
    image_ocr_checkpoint = os.path.join(checkpoint_dir, f"{sanitized_stem}_image_ocr.pkl")
    # Checkpoint for raw page data (list of tuples: (raw_markdown_text, images_dict))
    raw_page_data_checkpoint = os.path.join(checkpoint_dir, f"{sanitized_stem}_raw_page_data.pkl")

    # Step 1: Process PDF with OCR (with checkpoint)
    pdf_response = None
    load_msg = None
    if use_existing_checkpoints:
        pdf_response, load_msg = load_checkpoint(pdf_ocr_checkpoint) # Get message
        if load_msg:
            yield load_msg

    if pdf_response is None:
        msg = "🔍 正在處理 PDF OCR..."
        yield msg
        print(msg) # Console print
        pdf_response = process_pdf_with_mistral_ocr(pdf_path, mistral_client, model=ocr_model)
        save_msg = save_checkpoint(pdf_response, pdf_ocr_checkpoint) # save_checkpoint already prints
        if save_msg:
            yield save_msg
    else:
        print("ℹ️ 使用現有 PDF OCR 檢查點。")

    # Step 2: Process images with OCR (with checkpoint)
    ocr_by_page = {}
    if process_images:
        load_msg = None
        if use_existing_checkpoints:
            ocr_by_page, load_msg = load_checkpoint(image_ocr_checkpoint) # Get message
            if load_msg:
                yield load_msg

        if ocr_by_page is None or not ocr_by_page: # Check if empty dict from checkpoint or explicitly empty
            msg = f"🖼️ 正在使用 '{structure_model}' 處理圖片 OCR 與結構化..."
            yield msg
            print(msg) # Console print
            # Pass gemini_client and correct structure_model parameter name
            ocr_by_page = process_images_with_ocr(
                pdf_response, 
                mistral_client, 
                gemini_client, 
                openai_client, 
                structure_model=structure_model,
                structure_text_only=structure_text_only # Pass the text-only flag
            )
            save_msg = save_checkpoint(ocr_by_page, image_ocr_checkpoint) # save_checkpoint already prints
            if save_msg:
                yield save_msg
        else:
            print("ℹ️ 使用現有圖片 OCR 檢查點。")
    else:
        print("ℹ️ 跳過圖片 OCR 處理。") # process_images was False

    # Step 3: Create or load RAW page data (markdown text + image dicts)
    raw_page_data = None # List of tuples: (raw_markdown_text, images_dict)
    load_msg = None
    if use_existing_checkpoints:
        # Try loading the raw page data checkpoint
        raw_page_data, load_msg = load_checkpoint(raw_page_data_checkpoint)
        if load_msg:
            yield load_msg

    if raw_page_data is None:
        msg = "📝 正在建立原始頁面資料 (Markdown + 圖片資訊)..."
        yield msg
        print(msg)
        raw_page_data = []
        for page_idx, page in enumerate(pdf_response.pages):
            images_dict = {img.id: img.image_base64 for img in page.images}
            raw_md_text = page.markdown # Just the raw text with ![id](id)
            raw_page_data.append((raw_md_text, images_dict)) # Store as tuple

        # Save the RAW page data checkpoint
        save_msg = save_checkpoint(raw_page_data, raw_page_data_checkpoint)
        if save_msg:
            yield save_msg
    else:
        print("ℹ️ 使用現有原始頁面資料檢查點。")

    # Step 3.5: Conditionally insert image OCR results based on CURRENT UI selection
    pages_after_ocr_insertion = [] # List to hold markdown strings after potential OCR insertion
    if process_images and ocr_by_page: # Check if UI wants OCR AND if OCR results exist
        msg = "✍️ 根據目前設定，正在將圖片 OCR 結果插入 Markdown..."
        yield msg
        print(msg)
        for page_idx, (raw_md, _) in enumerate(raw_page_data): # Iterate through raw data
            # Insert OCR results into the raw markdown text BEFORE replacing links
            md_with_ocr = insert_ocr_below_images(raw_md, ocr_by_page, page_idx)
            pages_after_ocr_insertion.append(md_with_ocr)
    else:
        # If not inserting OCR, just use the raw markdown text
        if process_images and not ocr_by_page:
             msg = "ℹ️ 已勾選處理圖片 OCR，但無圖片 OCR 結果可插入 (可能需要重新執行圖片 OCR)。"
             yield msg
             print(msg)
        elif not process_images:
             msg = "ℹ️ 未勾選處理圖片 OCR，跳過插入步驟。"
             yield msg
             print(msg)
        # Use the raw markdown text directly
        pages_after_ocr_insertion = [raw_md for raw_md, _ in raw_page_data]

    # Step 3.6: Save images and replace links in the (potentially modified) markdown
    final_markdown_pages = [] # This list will have final file paths as links
    # Use sanitized_stem for image folder name
    image_folder_name = os.path.join(output_dir, f"images_{sanitized_stem}") 
    msg = f"🖼️ 正在儲存圖片並更新 Markdown 連結至 '{os.path.basename(image_folder_name)}'..."
    yield msg
    print(msg)
    # Iterate using the pages_after_ocr_insertion list and the original image dicts from raw_page_data
    for page_idx, (md_to_link, (_, images_dict)) in enumerate(zip(pages_after_ocr_insertion, raw_page_data)):
        # Now save images and replace links on the processed markdown (which might have OCR inserted)
        final_md = save_images_and_replace_links(md_to_link, images_dict, page_idx, image_folder=image_folder_name)
        final_markdown_pages.append(final_md)

    # Step 4: Translate the final markdown pages
    translated_markdown_pages = None # Initialize
    need_translation = "中文翻譯" in output_formats_selected
    if need_translation:
        # Translate the final list with correct image links, passing both clients
        translation_generator = translate_markdown_pages(
            final_markdown_pages, 
            gemini_client,
            openai_client, # Pass openai_client
            model=translation_model,
            system_instruction=translation_system_prompt
        )
        # Collect yielded pages from the translation generator
        translated_markdown_pages = [] # Initialize list to store results
        for item in translation_generator:
            # Check if it's a progress string or actual content/error
            # Simple check: assume non-empty strings starting with specific emojis are progress/status
            if isinstance(item, str) and (item.startswith("🔁") or item.startswith("⚠️") or item.startswith("✅")):
                 yield item # Forward progress/status string
            else:
                 # Assume it's translated content or an error marker page
                 translated_markdown_pages.append(item)
    else:
        yield "ℹ️ 跳過翻譯步驟 (未勾選中文翻譯)。"
        print("ℹ️ 跳過翻譯步驟 (未勾選中文翻譯)。")
        translated_markdown_pages = None # Ensure it's None if skipped

    # Step 5: Combine pages into complete markdown strings
    # The "original" output now correctly reflects the final state before translation
    final_markdown_original = "\n\n---\n\n".join(final_markdown_pages) # Use the final pages with links
    final_markdown_translated = "\n\n---\n\n".join(translated_markdown_pages) if translated_markdown_pages else None

    # Step 6: Save files based on selection - Use sanitized_stem
    saved_files = {}
    if "英文原文" in output_formats_selected:
        original_md_name = os.path.join(output_dir, f"{sanitized_stem}_original.md")
        try:
            with open(original_md_name, "w", encoding="utf-8") as f:
                f.write(final_markdown_original)
            msg = f"✅ 已儲存原文版：{original_md_name}"
            yield msg
            print(msg) # Console print
            saved_files["original_file"] = original_md_name
        except Exception as e:
            msg = f"❌ 儲存原文版失敗: {e}"
            yield msg
            print(msg)

    if "中文翻譯" in output_formats_selected and final_markdown_translated:
        translated_md_name = os.path.join(output_dir, f"{sanitized_stem}_translated.md")
        try:
            with open(translated_md_name, "w", encoding="utf-8") as f:
                f.write(final_markdown_translated)
            msg = f"✅ 已儲存翻譯版：{translated_md_name}"
            yield msg
            print(msg) # Console print
            saved_files["translated_file"] = translated_md_name
        except Exception as e:
            msg = f"❌ 儲存翻譯版失敗: {e}"
            yield msg
            print(msg)

    # Always report image folder path if images were processed/saved - Use sanitized_stem
    if process_images:
        image_folder_name = os.path.join(output_dir, f"images_{sanitized_stem}")
        msg = f"✅ 圖片資料夾：{image_folder_name}"
        yield msg
        print(msg) # Console print
        saved_files["image_folder"] = image_folder_name

    print(f"--- 完成處理檔案: {pdf_file.name} ---") # Console print

    # Return the final result dictionary for Gradio UI update
    yield {
        "saved_files": saved_files, # Dictionary of saved file paths
        "translated_content": final_markdown_translated,
        "original_content": final_markdown_original,
        "output_formats_selected": output_formats_selected # Pass back selections
    }

# ===== Gradio Interface =====

def create_gradio_interface():
    """Create a Gradio interface for the PDF to Markdown tool."""
    
    # Initialize clients
    load_dotenv()
    
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("❌ 找不到 MISTRAL_API_KEY，請檢查 .env 是否正確設置。")
    mistral_client = Mistral(api_key=mistral_api_key)
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("❌ 未在 .env 找到 GEMINI_API_KEY，請確認已正確設置。")
    gemini_client = genai.Client(api_key=gemini_api_key)

    # Initialize OpenAI client if library is available
    openai_client = None
    if OpenAI:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("⚠️ 未在 .env 找到 OPENAI_API_KEY。若要使用 OpenAI 模型，請設置此環境變數。")
            # Don't raise error, just disable OpenAI models if key is missing
        else:
            try:
                openai_client = OpenAI(api_key=openai_api_key)
                print("✅ OpenAI client initialized.")
            except Exception as e:
                print(f"❌ 初始化 OpenAI client 失敗: {e}")
                openai_client = None # Ensure client is None if init fails
    else:
        print("ℹ️ OpenAI library 未安裝，無法使用 OpenAI 模型。")

    # Define processing function for Gradio
    def process_pdf(
        pdf_file,
        ocr_model,
        structure_model,
        translation_model,
        translation_system_prompt,
        process_images,
        output_formats_selected, 
        output_dir,
        use_existing_checkpoints,
        structure_text_only # Added new parameter from Gradio input
    ):
        # Accumulate logs for console output
        log_accumulator = ""
        print("\n--- Gradio 處理請求開始 ---") # Console print
        # Placeholder for final markdown output
        final_result_content = "⏳ 等待處理結果..."

        # --- Early Exit Checks ---
        if pdf_file is None:
            log_accumulator += "❌ 請先上傳 PDF 檔案\n"
            print("❌ 錯誤：未上傳 PDF 檔案") # Console print
            # Final yield for error
            yield "錯誤：未上傳 PDF 檔案", log_accumulator
            return # Stop execution

        if not output_formats_selected:
             log_accumulator += "❌ 請至少選擇一種輸出格式（中文翻譯 或 英文原文）\n"
             print("❌ 錯誤：未選擇輸出格式") # Console print
             yield "錯誤：未選擇輸出格式", log_accumulator
             return # Stop execution

        pdf_path_obj = Path(pdf_file) # Use Path object for consistency
        filename_stem = pdf_path_obj.stem
        # Sanitize the filename stem (replace spaces with underscores)
        sanitized_stem = filename_stem.replace(" ", "_")
        print(f"收到檔案: {pdf_path_obj.name} (Sanitized Stem: {sanitized_stem})") # Console print
        print(f"選擇的輸出格式: {output_formats_selected}") # Console print

        # --- Output Directory Logic ---
        default_output_parent = os.path.join(os.path.expanduser("~"), "Desktop")
        default_output_folder = "MistralOCR_Output"

        if not output_dir or not output_dir.strip():
             # Default output_dir if empty or whitespace
            output_dir = os.path.join(default_output_parent, default_output_folder)
        # else: use the provided output_dir

        # Ensure output and checkpoint directories exist (use sanitized stem for checkpoint dir)
        checkpoint_dir = os.path.join(output_dir, f"checkpoints_{sanitized_stem}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
        except OSError as e:
            error_msg = f"❌ 無法建立目錄 '{output_dir}' 或 '{checkpoint_dir}': {e}"
            log_accumulator += f"{error_msg}\n"
            print(f"❌ 錯誤：{error_msg}") # Console print
            # Final yield for error
            yield f"錯誤：{error_msg}", log_accumulator
            return # Stop execution
        # --- End Output Directory Logic ---

        # --- Initial Log Messages ---
        # Print statements added within the block
        # Use yield with gr.update() for intermediate console updates
        log_accumulator += f"🚀 開始處理 PDF: {pdf_path_obj.name}\n"
        yield gr.update(), log_accumulator # Update only console
        log_accumulator += f"📂 輸出目錄: {output_dir}\n"
        yield gr.update(), log_accumulator # Update only console
        log_accumulator += f"💾 檢查點目錄: {checkpoint_dir}\n"
        yield gr.update(), log_accumulator # Update only console

        # Determine if translation is needed based on CheckboxGroup selection
        # The 'translate' checkbox is now less relevant, primary control is output_formats_selected
        need_translation_for_processing = "中文翻譯" in output_formats_selected
        log_accumulator += "✅ 將產生中文翻譯\n" if need_translation_for_processing else "ℹ️ 不產生中文翻譯 (未勾選)\n"
        yield gr.update(), log_accumulator # Update only console
        log_accumulator += "✅ 使用現有檢查點（如果存在）\n" if use_existing_checkpoints else "🔄 重新處理所有步驟（不使用現有檢查點）\n"
        yield gr.update(), log_accumulator # Update only console
        print(f"需要翻譯: {need_translation_for_processing}, 使用檢查點: {use_existing_checkpoints}") # Console print

        # --- Main Processing ---
        try:
            # process_pdf_to_markdown is a generator, iterate through its yields
            processor = process_pdf_to_markdown(
                pdf_path=pdf_file, # Pass the file path/object directly
                mistral_client=mistral_client,
                gemini_client=gemini_client,
                openai_client=openai_client, 
                ocr_model=ocr_model,
                structure_model=structure_model,
                structure_text_only=structure_text_only, # Pass text-only flag
                translation_model=translation_model,
                translation_system_prompt=translation_system_prompt if translation_system_prompt.strip() else None,
                process_images=process_images,
                # Removed duplicate process_images argument below
                output_formats_selected=output_formats_selected, # Pass selected formats
                output_dir=output_dir,
                checkpoint_dir=checkpoint_dir,
                use_existing_checkpoints=use_existing_checkpoints
            )

            result_data = None
            # Iterate through the generator from process_pdf_to_markdown
            for item in processor:
                if isinstance(item, dict): # Check if it's the final result dict
                    result_data = item
                    # Don't yield the dict itself to the console
                elif isinstance(item, str):
                    # Append and yield intermediate logs using gr.update()
                    log_accumulator += f"{item}\n"
                    yield gr.update(), log_accumulator # Update only console
                # Handle potential other types if necessary, otherwise ignore

            # --- Process Final Result for UI ---
            # This part runs after the processor generator is exhausted
            if result_data:
                final_log_message = "✅ 處理完成！"
                log_accumulator += f"{final_log_message}\n"
                print("--- Gradio 處理請求完成 ---") # Console print

                # Determine final_result_content based on selections in result_data
                selected_formats = result_data.get("output_formats_selected", [])
                original_content = result_data.get("original_content")
                translated_content = result_data.get("translated_content")

                content_parts = []
                if "英文原文" in selected_formats and original_content:
                    content_parts.append(f"# 英文原文\n\n{original_content}")
                if "中文翻譯" in selected_formats and translated_content:
                     content_parts.append(f"# 中文翻譯\n\n{translated_content}")

                if content_parts:
                    final_result_content = "\n\n---\n\n".join(content_parts)
                else:
                    final_result_content = "ℹ️ 未選擇輸出格式或無內容可顯示。"

            else:
                 final_log_message = "⚠️ 處理完成，但未收到預期的結果字典。"
                 log_accumulator += f"{final_log_message}\n"
                 print(f"⚠️ 警告：{final_log_message}") # Console print
                 final_result_content = "❌ 處理未完成或未產生預期輸出。"

            # Final yield: provide values for BOTH outputs
            yield final_result_content, log_accumulator

        except Exception as e:
            error_message = f"❌ Gradio 處理過程中發生未預期錯誤: {str(e)}"
            log_accumulator += f"{error_message}\n"
            print(f"❌ 嚴重錯誤：{error_message}") # Console print
            import traceback
            traceback.print_exc() # Print full traceback to console
            # Final yield in case of error: provide values for BOTH outputs
            yield error_message, log_accumulator

    # Create Gradio interface
    with gr.Blocks(title="Mistral OCR 翻譯工具") as demo: # Updated title slightly
        gr.Markdown("# Mistral OCR 翻譯PDF轉Markdown格式工具")
        gr.Markdown("將 PDF 文件轉為 Markdown 格式，支援圖片 OCR 和英文到繁體中文翻譯，使用 **Mistral**、**Gemini** 和 **OpenAI** 模型。") # Added OpenAI
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="上傳 PDF 檔案", file_types=[".pdf"])
                
                with gr.Accordion("基本設定", open=True):
                    # Define default path for placeholder clarity
                    default_output_path_display = os.path.join("桌面", "MistralOCR_Output") # Simplified for display
                    output_dir = gr.Textbox(
                        label="輸出目錄 (請貼上完整路徑)",
                        placeholder=f"留空預設儲存至：{default_output_path_display}",
                        info="將所有輸出檔案 (Markdown, 圖片, 檢查點) 儲存於此目錄。",
                        value="" # Default logic remains in process_pdf
                    )

                    use_existing_checkpoints = gr.Checkbox(
                        label="使用現有檢查點（如果存在）", 
                        value=True,
                        info="啟用後，如果檢查點存在，將跳過已完成的步驟。"
                    )

                    output_format = gr.CheckboxGroup(
                        label="輸出格式 (可多選)",
                        choices=["中文翻譯", "英文原文"],
                        value=["中文翻譯", "英文原文"], # Default to both
                        info="選擇您需要儲存的 Markdown 檔案格式。"
                    )

                with gr.Accordion("處理選項", open=True):
                    process_images = gr.Checkbox(
                        label="處理圖片 OCR", 
                        value=True,
                        info="啟用後，將對 PDF 中的圖片進行 OCR 辨識"
                    )
                    
                    # The 'translate' checkbox is now redundant as format selection controls translation
                    # We can hide or remove it. Let's comment it out for now.
                    # translate = gr.Checkbox(
                    #     label="翻譯成中文",
                    #     value=True,
                    #     info="啟用後，將英文內容翻譯為中文 (若輸出格式已選中文翻譯則自動啟用)"
                    # )

                with gr.Accordion("模型設定", open=False):
                    ocr_model = gr.Dropdown(
                        label="OCR 模型", 
                        choices=["mistral-ocr-latest"], 
                        value="mistral-ocr-latest"
                    )
                    structure_model = gr.Dropdown(
                        label="結構化模型 (用於圖片 OCR)", 
                        choices=[
                            ("pixtral-12b-latest (Recommended)", "pixtral-12b-latest"),
                            ("gemini-2.0-flash (Recommended)", "gemini-2.0-flash"),
                            ("gpt-4o-mini", "gpt-4o-mini"),
                            ("gpt-4o", "gpt-4o"),
                            ("gpt-4.1-nano (Not Recommended)", "gpt-4.1-nano"),
                            ("gpt-4.1-mini", "gpt-4.1-mini"),
                            ("gpt-4.1", "gpt-4.1")
                        ], 
                        value="gemini-2.0-flash",
                        info="選擇用於結構化圖片 OCR 結果的模型。選擇 Gemini 或 OpenAI 模型需要對應的 API Key 在 .env 檔案中設定。"
                    )
                    structure_text_only = gr.Checkbox(
                        label="僅用文字進行結構化 (節省 Token)",
                        value=False,
                        info="勾選後，僅將圖片的初步 OCR 文字傳送給 Gemini 或 OpenAI 進行結構化，不傳送圖片本身。對 Pixtral 無效。⚠️注意：缺少圖片視覺資訊可能導致結構化效果不佳，建議僅在 OCR 文字已足夠清晰時使用。"
                    )
                    translation_model = gr.Dropdown(
                        label="翻譯模型", 
                        choices=[
                            ("gemini-2.0-flash (Recommended)", "gemini-2.0-flash"), 
                            ("gemini-2.5-pro-exp-03-25", "gemini-2.5-pro-exp-03-25"), 
                            ("gemini-2.0-flash-lite", "gemini-2.0-flash-lite"),
                            ("gpt-4o", "gpt-4o"), 
                            ("gpt-4o-mini", "gpt-4o-mini"),
                            ("gpt-4.1-nano (Not Recommended)", "gpt-4.1-nano"),
                            ("gpt-4.1-mini", "gpt-4.1-mini"),
                            ("gpt-4.1", "gpt-4.1")
                        ], 
                        value="gemini-2.0-flash",
                        info="選擇用於翻譯的模型。選擇 Gemini 或 OpenAI 模型需要對應的 API Key 在 .env 檔案中設定。"
                    )
                with gr.Accordion("進階設定", open=False):
                    translation_system_prompt = gr.Textbox(
                        label="翻譯系統提示詞", 
                        value=DEFAULT_TRANSLATION_SYSTEM_INSTRUCTION,
                        lines=10
                    )
                
                process_button = gr.Button("開始處理", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Tab("輸出結果"):
                    output = gr.Markdown(label="輸出結果")
                
                with gr.Tab("處理日誌"):
                    console_output = gr.Textbox(
                        label="處理進度", 
                        lines=20,
                        max_lines=50,
                        interactive=False,
                        autoscroll=True # Add autoscroll
                    )

                with gr.Tab("使用說明"):  
                    
                    gr.Markdown("""
                        # 使用說明（本地版本）

                        1. 上傳 PDF 檔案（可拖曳或點擊上傳）  
                        2. 基本設定：  
                        - 指定輸出目錄（可選，留空使用預設目錄）  
                        - 選擇是否使用現有檢查點（如果存在）  
                        - 選擇輸出格式（中文翻譯、英文原文）  
                        3. 處理選項：  
                        - 選擇是否處理圖片 OCR  
                        - 選擇是否翻譯成中文（若輸出格式僅選「英文原文」則略過翻譯）  
                        4. 模型與進階設定（可選）：  
                        - 選擇 OCR、結構化、翻譯模型  
                        - 修改翻譯提示詞（若需其他語言）  
                        5. 點擊「開始處理」按鈕  
                        6. 處理期間可於「處理日誌」查看進度  
                        7. 完成後，結果將顯示於「輸出結果」頁，並自動儲存至指定目錄  

                        ## API 金鑰設定 (.env)

                        請於專案根目錄建立 `.env` 檔案，填入以下內容：

                        ```
                        MISTRAL_API_KEY=your_mistral_key
                        GEMINI_API_KEY=your_gemini_key       # 可選
                        OPENAI_API_KEY=your_openai_key       # 可選
                        ```

                        ## 檢查點說明

                        - **PDF OCR 檢查點**：儲存 PDF 的文字識別結果  
                        - **圖片 OCR 檢查點**：儲存圖片區塊的 OCR 結果  
                        - **Markdown 檢查點**：儲存已產出的 Markdown 檔案  
                        可取消勾選「使用現有檢查點」重新處理，或手動刪除 `checkpoints/` 資料夾。

                        ## 輸出檔案

                        - `[檔名]_translated.md`：翻譯後的 Markdown 檔案  
                        - `[檔名]_original.md`：原始英文 Markdown 檔案  
                        - `images_[檔名]/`：提取圖片資料夾  
                        - `checkpoints/`：處理過程中的中繼檔案  
                    """)

                

        # Define outputs for the click event
        # The order matches how Gradio handles generators:
        # Last yield goes to the first output, intermediate yields go to the second.
        outputs_list = [output, console_output]

        # Define inputs for the click event (remove console_output)
        inputs_list=[
            pdf_file,
            ocr_model,
            structure_model,
            translation_model,
            translation_system_prompt,
            process_images,
            # translate, # Removed from inputs as it's redundant now
            output_format, # Now CheckboxGroup list
            output_dir,
            use_existing_checkpoints,
            structure_text_only # Added new checkbox input
        ]

        # Use process_button.click with the generator function
        process_button.click(
            fn=process_pdf,
            inputs=inputs_list,
            outputs=outputs_list
        )

        # Add event handler to exit script when UI is closed/unloaded
        # Removed inputs and outputs arguments as they are not accepted by unload
        # demo.unload(fn=lambda: os._exit(0))


        gr.Markdown(""" 
            ---

            **免責聲明**  
            本工具僅供學習與研究用途，整合 Mistral、Google Gemini 和 OpenAI API。請確保：
            - 您擁有合法的 API 金鑰，並遵守各服務條款（[Mistral](https://mistral.ai/terms)、[Gemini](https://ai.google.dev/terms)、[OpenAI](https://openai.com/policies)）。  
            - 上傳的 PDF 文件符合版權法規，您有權進行處理。  
            - 翻譯結果可能有誤，請自行驗證。  
            本工具不儲存任何上傳檔案或 API 金鑰，所有處理均在暫存環境中完成。

            **版權資訊**  
            Copyright © 2025 David Chang. 根據 MIT 授權發布，詳見 [LICENSE](https://github.com/dodo13114arch/mistralocr-pdf2md-translator/blob/main/LICENSE)。  
            GitHub: https://github.com/dodo13114arch/mistralocr-pdf2md-translator
            """)
    
    return demo

# ===== Main Execution =====

if __name__ == "__main__":
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch()
