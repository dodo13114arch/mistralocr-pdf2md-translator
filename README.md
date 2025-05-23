# Mistral OCR 翻譯PDF轉Markdown格式工具

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/dodo13114arch/mistral-ocr-translator-demo) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dodo13114arch/mistralocr-pdf2md-translator)

**[點此試用線上 Demo (Hugging Face Spaces)](https://huggingface.co/spaces/dodo13114arch/mistral-ocr-translator-demo)** 

---

本工具可將 PDF 文件自動化轉換為 Markdown 格式，包含以下功能：

1. 使用 **Mistral OCR** 模型辨識 PDF 內文與圖片。
2. 將辨識結果組成含圖片的 Markdown 檔。
3. 使用 **Gemini** 或 **OpenAI** 模型將英文內容翻譯為**台灣繁體中文** (或其他語言，透過修改提示詞)。
4. 使用 **Pixtral**, **Gemini**, 或 **OpenAI** 模型對圖片中的內容進行結構化 OCR (可選)。
5. 匯出 Markdown 檔（原文版 + 翻譯版）與對應圖片。

⚠ 注意： 本專案依賴於 httpx 和 uvicorn，這些套件目前尚未支援 h11 版本 0.16.0 或以上。雖然 h11 在 0.16.0 版本中修復了已知的安全性漏洞（CVE-2025-43859），但目前的依賴組合可能仍使用較舊的 h11 版本。使用者可自行調整依賴版本，但需自行承擔可能的風險。(2025,5,24 update) 

## 安裝需求

1. 安裝必要的 Python 套件：

```bash
pip install -r requirements.txt
```

2. 建立 `.env` 檔案，並設定 API 金鑰：

```
# Mistral AI API Key
# 請從 https://console.mistral.ai/ 獲取
MISTRAL_API_KEY=your_mistral_api_key_here

# Google Gemini API Key (若需使用 Gemini 模型則必要)
# 請從 https://aistudio.google.com/app/apikey 獲取
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI API Key (若需使用 OpenAI 模型則必要)
# 請從 https://platform.openai.com/api-keys 獲取
OPENAI_API_KEY=your_openai_api_key_here
```

## 使用方法

### 使用 Gradio 介面 

執行以下命令啟動 Gradio 介面：

```bash
python mistralocr_app.py
```

然後在瀏覽器中開啟顯示的網址（通常是 http://127.0.0.1:7860）。

### 介面使用說明

1.  上傳 PDF 檔案（可拖曳或點擊上傳）
2.  基本設定：
    *   指定輸出目錄（可選，留空將儲存至桌面 `MistralOCR_Output` 資料夾）
    *   選擇是否使用現有檢查點（預設啟用）
    *   選擇輸出格式（**可複選**「中文翻譯」、「英文原文」，預設兩者皆選）
3.  處理選項：
    *   選擇是否處理圖片 OCR（預設啟用）
    *   *(翻譯功能由「輸出格式」選項控制)*
4.  模型設定（可選）：
    *   選擇 OCR 模型 (目前僅支援 Mistral)。
    *   選擇結構化模型 (Pixtral, Gemini, OpenAI)。
    *   選擇翻譯模型 (Gemini, OpenAI)。
5.  進階設定（可選）：
    *   修改翻譯系統提示詞 (可設定調整翻譯成其他語言)
6.  點擊「開始處理」按鈕
7.  處理過程中，可在「處理日誌」標籤頁查看詳細進度
8.  處理完成後，結果將顯示在「輸出結果」標籤頁，檔案會自動儲存到指定目錄

## 檔案說明

- `mistralocr_app.py`：主要的 Gradio 應用程式腳本
- `requirements.txt`：執行所需的 Python 套件列表
- `mistralocr_pdf2md.ipynb`：根據 [Mistral 官方 Notebook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb) 修改的學習筆記，加入翻譯與圖片儲存功能。
- `mistralocr_pdf2md_claude_refined.ipynb`: (可選) 另一個開發/測試用的 Notebook 版本。
- **輸出檔案 (位於指定的輸出目錄內):**
    *   `[檔名]_original.md`：處理後的原始（英文）Markdown 檔案 (若有勾選)
    *   `[檔名]_translated.md`：翻譯後的（繁體中文）Markdown 檔案 (若有勾選)
    *   `images_[檔名]/`：從 PDF 擷取並儲存的圖片資料夾
    *   `checkpoints_[檔名]/`：處理過程中的檢查點資料夾，包含中間處理結果

## 注意事項

- 處理過程中會在**輸出目錄**內建立 `checkpoints_[檔名]` 資料夾，儲存中間結果，以便在中斷後繼續處理，避免重複請求 API。若要強制重新處理，可取消勾選「使用現有檢查點」或刪除對應的檢查點資料夾。
- 擷取的圖片會儲存在**輸出目錄**內的 `images_[檔名]` 資料夾中。
- 最終的 Markdown 檔案 (`_original.md`, `_translated.md`) 會儲存在使用者指定的**輸出目錄**中（若未指定，則預設為桌面上的 `MistralOCR_Output` 資料夾）。
- 請確保 `.env` 檔案已正確設定您的 Mistral AI 和 Google Gemini API 金鑰。

## 技術來源與引用

本專案整合並改作自以下技術或官方範例：

- [Mistral 文件處理功能說明](https://docs.mistral.ai/capabilities/document/)
- [Mistral 官方 Colab Notebook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/ocr/structured_ocr.ipynb)

其中 `mistralocr_pdf2md.ipynb` 為學習用途所建立，根據官方範例進行延伸與修改，加入了中英文翻譯流程與本地圖片儲存功能等。

本工具亦整合以下第三方 API/工具：

- [Mistral API](https://mistral.ai/)
- [Google Gemini API](https://ai.google.dev/)
- [OpenAI API](https://openai.com/)
- [Gradio](https://www.gradio.app/)

> 本專案為個人開發與學習用途，與上述服務提供者無任何官方關聯。請使用者自行準備合法的 API 金鑰，並遵守各 API 提供商的使用條款 ([Mistral](https://mistral.ai/terms)、[Gemini](https://ai.google.dev/terms)、[OpenAI](https://openai.com/policies))。
