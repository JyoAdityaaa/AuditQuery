// app.js - Vanilla JS Frontend Controllers

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const docList = document.getElementById("doc-list");
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadForm = document.getElementById("upload-form");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadStatus = document.getElementById("upload-status");
  
  const queryForm = document.getElementById("query-form");
  const queryInput = document.getElementById("query-input");
  const queryBtn = document.getElementById("query-btn");
  const topKSelect = document.getElementById("top-k");
  
  const responsePanel = document.getElementById("response-panel");
  const answerText = document.getElementById("answer-text");
  const citationsList = document.getElementById("citations-list");

  let selectedFiles = [];

  // 1. Initial Load: Fetch active indexed documents
  loadIndexedDocuments();

  async function loadIndexedDocuments() {
    try {
      const response = await fetch("/api/documents");
      if (!response.ok) {
        throw new Error(await getErrorDetail(response));
      }
      const data = await response.json();
      renderDocumentList(data.documents);
    } catch (err) {
      console.error("Failed to load documents:", err);
      showUploadStatus(`Failed to load knowledge base: ${err.message}`, "error");
    }
  }

  function renderDocumentList(documents) {
    docList.innerHTML = "";
    if (!documents || documents.length === 0) {
      docList.innerHTML = '<li class="doc-placeholder">No documents indexed yet</li>';
      return;
    }

    documents.forEach(doc => {
      const li = document.createElement("li");
      li.className = "doc-item";
      li.innerHTML = `
        <span>📄 ${doc.filename}</span>
        <span class="meta-val" style="font-size: 0.7rem; font-weight:600;">Indexed</span>
      `;
      docList.appendChild(li);
    });
  }

  // 2. File Select & Drag-and-Drop Handlers
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFilesSelected(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFilesSelected(e.target.files);
    }
  });

  function handleFilesSelected(files) {
    selectedFiles = Array.from(files).filter(file => file.type === "application/pdf");
    
    if (selectedFiles.length === 0) {
      showUploadStatus("Please select valid PDF documents.", "error");
      uploadBtn.disabled = true;
      dropZone.querySelector(".drop-text").textContent = "Click or drag PDFs here";
      return;
    }

    // Update Drop Zone UI
    const fileNames = selectedFiles.map(f => f.name).join(", ");
    dropZone.querySelector(".drop-text").textContent = `${selectedFiles.length} file(s) selected: ${fileNames}`;
    uploadBtn.disabled = false;
    showUploadStatus("", "info"); // Clear status
  }

  // 3. Document Indexing Form Handler
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0) return;

    // UI Loading State
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Indexing...";
    showUploadStatus("Extracting chunks and generating CPU embeddings...", "info");

    const formData = new FormData();
    selectedFiles.forEach(file => {
      formData.append("files", file);
    });

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        throw new Error(await getErrorDetail(response));
      }

      const result = await response.json();
      
      const added = result.added || [];
      const skipped = result.skipped || [];
      const removed = result.removed || [];
      
      let msg = "Synchronization complete! ";
      if (added.length > 0) msg += `Indexed: ${added.join(", ")}. `;
      if (skipped.length > 0) msg += `Unchanged: ${skipped.join(", ")}. `;
      
      showUploadStatus(msg, "success");
      
      // Reset inputs
      selectedFiles = [];
      fileInput.value = "";
      dropZone.querySelector(".drop-text").textContent = "Click or drag PDFs here";
      
      // Reload Document Registry
      await loadIndexedDocuments();

    } catch (err) {
      showUploadStatus(err.message, "error");
      uploadBtn.disabled = false;
    } finally {
      uploadBtn.textContent = "Index & Sync Files";
    }
  });

  // 4. Q&A Query Form Handler
  queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    // UI Loading State
    queryBtn.disabled = true;
    queryBtn.textContent = "Simplifying...";
    responsePanel.style.display = "none";

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: query,
          k: parseInt(topKSelect.value, 10)
        })
      });

      if (!response.ok) {
        throw new Error(await getErrorDetail(response));
      }

      const data = await response.json();
      renderQueryResponse(data);

    } catch (err) {
      alert(`Query Failed: ${err.message}`);
    } finally {
      queryBtn.disabled = false;
      queryBtn.textContent = "Ask AuditQuery";
    }
  });

  function renderQueryResponse(data) {
    // 1. Render Markdown answer — parse then sanitize before inserting as HTML
    const rawMarkdown = data.answer || "";
    const parsedHtml = marked.parse(rawMarkdown);
    answerText.innerHTML = DOMPurify.sanitize(parsedHtml);

    // 2. Render Citations List
    citationsList.innerHTML = "";
    const sources = data.sources || [];
    
    if (sources.length === 0) {
      citationsList.innerHTML = '<p class="doc-placeholder">No matching context fragments were retrieved.</p>';
    } else {
      sources.forEach((src, idx) => {
        const item = document.createElement("div");
        item.className = "citation-item";
        // Escape the chunk text to prevent XSS from document content
        const safeChunk = DOMPurify.sanitize(src.chunk_text);
        item.innerHTML = `
          <div class="citation-meta">
            <span>Reference: <span class="meta-val">#${idx + 1}</span></span>
            <span>Source: <span class="meta-val">📄 ${src.source}</span></span>
            <span>Page: <span class="meta-val">${src.page}</span></span>
            <span>Match: <span class="meta-score">${(src.similarity * 100).toFixed(1)}%</span></span>
          </div>
          <div class="citation-text">${safeChunk}</div>
        `;
        citationsList.appendChild(item);
      });
    }

    // 3. Reveal Result Panels
    responsePanel.style.display = "block";
    responsePanel.scrollIntoView({ behavior: "smooth" });
  }

  // Helper: Read server API error responses
  async function getErrorDetail(response) {
    try {
      const errData = await response.json();
      return errData.detail || response.statusText;
    } catch {
      return response.statusText;
    }
  }

  // Helper: display file upload status banners
  function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = "status-msg"; // reset
    if (message) {
      uploadStatus.classList.add(type);
    }
  }
});
