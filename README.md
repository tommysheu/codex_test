# GPT-4o Chatbot 網頁

此專案提供一個使用 FastAPI 建立的後端與 HTML/JavaScript 前端的聊天機器人示例。後端透過 GPT-4o API 回答問題，並支援四種使用模式：

1. **純 LLM 模式**：直接以模型的既有知識回覆。
2. **網頁查詢（DuckDuckGo） + LLM 模式**：透過 `duckduckgo-search` 取得即時網頁摘要，再交由 GPT-4o 產生結合查詢資訊的回答。
3. **網頁查詢（SerpAPI） + LLM 模式**：利用 SerpAPI 的 Google Search 結果，讓 GPT-4o 擷取與引用最新的網頁內容。
4. **網頁查詢（自動判斷） + LLM 模式**：系統根據提問是否具有時效性或即時資訊需求決定是否啟動搜尋；若需要則呼叫 SerpAPI 擷取最新資料，否則僅以 LLM 回覆。

## 專案結構

```
.
├── app
│   └── main.py          # FastAPI 服務
├── frontend
│   └── index.html       # 前端單頁應用程式
├── requirements.txt     # 相依套件
└── README.md            # 操作說明
```

## 前置需求

- Python 3.10+
- OpenAI API 金鑰，並設定環境變數：

```bash
export OPENAI_API_KEY="your-openai-key"
```

> 若未設定金鑰，後端將回報 `OPENAI_API_KEY is not configured on the server.` 錯誤。

- （選用）SerpAPI 金鑰，若要啟用 SerpAPI 模式需設定：

```bash
export SERPAPI_API_KEY="your-serpapi-key"
```

> 若未設定金鑰，選擇 SerpAPI 模式時後端將返回錯誤提示。

## 安裝與啟動

1. 安裝套件：

   ```bash
   pip install -r requirements.txt
   ```

2. 啟動 FastAPI 伺服器（預設在 `http://127.0.0.1:8000`）：

   ```bash
   uvicorn app.main:app --reload
   ```

3. 在瀏覽器開啟 `http://127.0.0.1:8000/` 使用聊天介面。

4. FastAPI 內建 Swagger 文件可在 `http://127.0.0.1:8000/docs` 查看與測試 API。

## API 端點

- `POST /chat`
  - **body**
    ```json
    {
      "message": "使用者訊息",
      "mode": "llm_only" | "search_duckduckgo" | "search_serpapi" | "search_auto",
      "history": [
        {"role": "user", "content": "前一輪的提問"},
        {"role": "assistant", "content": "前一輪的回答"}
      ]
    }
    ```
  - **response**：以 [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events) 形式串流傳回，事件類型說明如下：
      - `meta`：包含 `mode`、`used_search`、`search_provider` 與 `search_results` 等欄位；若為自動模式還會回傳 `auto_search_triggered` 表示是否實際執行 SerpAPI 搜尋。
    - `delta`：逐步輸出 `text` 欄位，為模型即時生成的片段。
    - `done`：回傳最終完整的 `text` 內容。
    - `error`：若產生錯誤則附上 `message` 描述。
    
    範例：

    ```text
    event: meta
    data: {"mode": "search_duckduckgo", "used_search": true, "search_provider": "duckduckgo", "search_results": [{"title": "結果標題", "snippet": "摘要", "url": "https://..."}]}

    event: delta
    data: {"text": "第一段回覆"}

    event: done
    data: {"text": "完整回答"}
    ```

## 注意事項

- DuckDuckGo 搜尋由 `duckduckgo-search` 套件驅動，會先以台灣地區（`tw-tzh`）查詢，若無結果再以全球設定重試。若網路連線失敗，後端仍會嘗試以 LLM 回覆。
- SerpAPI 搜尋使用 `google-search-results` 套件，需提供有效的 SerpAPI API 金鑰，回傳的 `search_provider` 會標示為 `serpapi`。在「網頁查詢（自動判斷）」模式下，系統會根據關鍵字與語意判斷是否需要啟動即時搜尋，若啟用則同樣以 SerpAPI 提供內容。
- 前端預設顯示繁體中文 UI，GPT-4o 會自動依照上下文在中英雙語間轉換。

## 開發建議

- 若要擴充更多資料來源，可在 `app/main.py` 中新增對應的查詢函式並在 `/chat` 端點分支處理模式。
- 可將前端改寫為框架式應用（例如 React/Vue）並透過 FastAPI 的 static mount 部署。此範例以簡潔的原生 HTML/JS 示範核心流程。
