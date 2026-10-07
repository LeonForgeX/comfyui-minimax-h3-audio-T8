# 採樣前準備效能：2026-10-06

針對 Leon v8.39.1 的採樣前等待，只減少同一安全檢查邊界內重複的內容身份計算。採樣步數、sigma、seed、CFG、上下文、VAE 編碼、3D 放大器及 LOW/HIGH 數值路徑不變。這是配對原始碼更新，Registry 版本保留 1.86.0。

## 草稿

內部一採草稿以同步 bind-and-load 操作避免建立身份後立即重做同一檢查。只有經認證的原生唯讀 getter、無副作用的 state_dict 路徑及已知原生 bypass 注入可走此路徑。不同 LOW/HIGH patcher 仍分別建立身份；相同 patcher 僅在認證安全時共用完整描述。未知 getter、hooks、clone 回呼或注入組合保留原有完整驗證。

公開 bind_low/load_low 仍是兩個獨立邊界；公開讀取必須重新驗證。快取命中讀取後、保存前、後續採樣邊界的內容檢查保留。沒有跨執行的 tensor version、檔案 stat 或 storage pointer 身份捷徑。實現身份包含本輪程式碼，更新後舊草稿可能不相容；首次更新建議完整生成新草稿。

## 外部續拍上下文

新增 validate_external_rgb_pcm_contexts，在同一次群組檢查中先逐一驗證所有來源、holder、metadata 和 tensor，再讓安全的原生 video/audio VAE 共用一次完整 producer 描述。下一次檢查重新讀取實際權重。每個 encode 後的初始上下文描述仍獨立建立，不跨 LOW/HIGH encode 共用身份。自訂 wrapper、getter、state_dict hooks、非認證注入或未知 producer 保留逐一校验。單一 context API 保留。

Leon 必須與本次 T8 原始碼一起更新；缺少群組 helper 時 preflight 會明確要求配對更新。未動舊工作流、節點 ID/schema、既有 cache receipt 完整性檢查或接縫已接受樣例。

## 驗證範圍

固定官方 Core b5cc8830279eae909a59de030af1e50761c36751 的完整 22 檔 native CPU gate 為 605 passed／27 skipped（112.71 秒）。草稿新增 27 項 focused、外部上下文 60 項 focused 通過；涵蓋實際 tiny native Euler、reduced random VAE、快取讀取後重新驗證、.data／NaN／LoRA／conditioning／來源變更及未知 getter／Tensor fallback。GitHub exact-head CI 另以實際工作記錄為準。沒有 Windows 真實 GPU 工作流、整片品質、顯存或實際加速秒數的測量；GPU speed = NOT_RUN。

控制流程計數：安全原生不同 LOW/HIGH clone 在首次生成／快取未命中時，每路初始身份 capture 由 2／3 次降至 1 次，快取命中時每路 4→2，保留讀取後校驗。安全原生外部上下文每次 pair guard 每個 VAE 完整 producer 掃描 2→1；prepare 4→3；Leon prepare 含結尾 adapter guard 為 6→4。VAE encode 次數保持 video 2／audio 1。這些是測試中的呼叫數，並非實際 GPU 啟動秒數。
