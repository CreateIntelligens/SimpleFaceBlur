import React, { useState, useRef, ChangeEvent } from 'react';
import axios from 'axios';
import './App.css';

interface Face {
  id: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  area: number;
  selected: boolean;
}

const App: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [originalFile, setOriginalFile] = useState<File | null>(null);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [faces, setFaces] = useState<Face[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [tool, setTool] = useState<'pen' | 'eraser'>('pen');
  const [blurMode, setBlurMode] = useState<'emoji' | 'blur'>('emoji');
  const [emoji, setEmoji] = useState('😊');
  const [darkMode, setDarkMode] = useState(true);
  const [batchProgress, setBatchProgress] = useState<{ current: number; total: number; filename: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);

  const handleUpload = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setOriginalFile(file);
      setImage(URL.createObjectURL(file));
      setPreviewImage(null);
      setProcessedImage(null);
      setFaces([]);
      setStatus('');
    }
  };

  const fetchPreviewWithFaces = async (currentFaces: Face[]) => {
    if (!originalFile) return;

    const selectedIds = currentFaces.filter(f => f.selected).map(f => f.id);

    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('selected_ids', JSON.stringify(selectedIds));
    formData.append('mode', blurMode);
    formData.append('emoji', emoji);

    try {
      const res = await axios.post('/api/preview', formData, { responseType: 'blob' });
      setPreviewImage(URL.createObjectURL(res.data));
    } catch (err) {
      console.error('Preview failed:', err);
    }
  };

  const detectFaces = async () => {
    if (!originalFile) return;
    setLoading(true);
    setStatus('檢測中...');

    const formData = new FormData();
    formData.append('image', originalFile);

    try {
      const res = await axios.post('/api/detect', formData);
      const detectedFaces = res.data.faces.map((f: any) => ({
        ...f,
        selected: true
      }));
      setFaces(detectedFaces);
      setProcessedImage(null);
      setStatus(`檢測完成 - 發現 ${detectedFaces.length} 個人臉（已全選）`);

      await fetchPreviewWithFaces(detectedFaces);
    } catch (err) {
      setStatus('檢測失敗');
    } finally {
      setLoading(false);
    }
  };

  const toggleFace = async (id: number) => {
    const newFaces = faces.map(f => {
      if (f.id === id) {
        return { ...f, selected: tool === 'pen' };
      }
      return f;
    });
    setFaces(newFaces);
    await fetchPreviewWithFaces(newFaces);
  };

  const selectAll = async () => {
    const newFaces = faces.map(f => ({ ...f, selected: true }));
    setFaces(newFaces);
    await fetchPreviewWithFaces(newFaces);
    setStatus(`已全選 ${newFaces.length} 個人臉`);
  };

  const selectNone = async () => {
    const newFaces = faces.map(f => ({ ...f, selected: false }));
    setFaces(newFaces);
    await fetchPreviewWithFaces(newFaces);
    setStatus('已取消所有選擇');
  };

  const viewSelection = async () => {
    if (faces.length === 0) {
      setStatus('請先檢測人臉');
      return;
    }
    await fetchPreviewWithFaces(faces);
    const selected = faces.filter(f => f.selected).length;
    setStatus(`查看選擇: ${selected}/${faces.length} 個人臉`);
  };

  const executeBlur = async () => {
    if (!originalFile || faces.length === 0) return;
    setLoading(true);
    setStatus('處理中...');

    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('faces', JSON.stringify(faces.filter(f => f.selected)));
    formData.append('mode', blurMode);
    formData.append('emoji', emoji);

    try {
      const res = await axios.post('/api/blur', formData, { responseType: 'blob' });
      setProcessedImage(URL.createObjectURL(res.data));
      setPreviewImage(null);
      const selected = faces.filter(f => f.selected).length;
      setStatus(`遮蔽完成 - 已遮蔽 ${selected} 個人臉`);
    } catch (err) {
      setStatus('處理失敗');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileList = Array.from(files);
    const confirmed = window.confirm(
      `即將批次處理 ${fileList.length} 張圖片\n\n` +
      `⚠️ 警告：批次模式會自動遮蔽所有檢測到的人臉\n` +
      `處理後的圖片將自動下載\n\n` +
      `確定要繼續嗎？`
    );

    if (!confirmed) {
      e.target.value = '';
      return;
    }

    setLoading(true);
    let successCount = 0;

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      setBatchProgress({ current: i + 1, total: fileList.length, filename: file.name });
      setStatus(`批次處理中: ${i + 1}/${fileList.length} - ${file.name}`);

      try {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('mode', blurMode);
        formData.append('emoji', emoji);

        const res = await axios.post('/api/process', formData, { responseType: 'blob' });

        // 自動下載
        const url = URL.createObjectURL(res.data);
        const link = document.createElement('a');
        link.href = url;
        const name = file.name.replace(/\.[^/.]+$/, '');
        link.download = `${name}_blurred.jpg`;
        link.click();
        URL.revokeObjectURL(url);

        successCount++;
      } catch (err) {
        console.error(`Failed to process ${file.name}:`, err);
      }
    }

    setBatchProgress(null);
    setLoading(false);
    setStatus(`批次處理完成 - 成功 ${successCount}/${fileList.length}`);
    e.target.value = '';
  };

  const saveResult = () => {
    if (processedImage) {
      const link = document.createElement('a');
      link.href = processedImage;
      link.download = `blurred_${originalFile?.name || 'image.jpg'}`;
      link.click();
    }
  };

  const handleImageClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!faces.length || processedImage) return;

    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();

    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;

    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    const clickX = relX * naturalWidth;
    const clickY = relY * naturalHeight;

    for (const face of faces) {
      if (clickX >= face.x1 && clickX <= face.x2 &&
        clickY >= face.y1 && clickY <= face.y2) {
        await toggleFace(face.id);
        break;
      }
    }
  };

  const selectedCount = faces.filter(f => f.selected).length;
  const displayImage = processedImage || previewImage || image;

  return (
    <div className={`app ${darkMode ? 'dark' : 'light'}`}>
      <header className="header">
        <h1>😊 人臉遮蔽工具</h1>
        <label className="theme-toggle">
          <input type="checkbox" checked={darkMode} onChange={() => setDarkMode(!darkMode)} />
          <span>深色模式</span>
        </label>
      </header>

      <main className="main">
        <section className="preview-section">
          <div className="toolbar">
            <button className="btn btn-upload" onClick={() => fileInputRef.current?.click()}>
              📁 上傳圖片
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleUpload} hidden />

            <button className="btn btn-detect" onClick={detectFaces} disabled={!image || loading}>
              🔍 檢測人臉
            </button>
          </div>

          <div className="tool-row">
            <span>選擇工具：</span>
            <button
              className={`btn btn-tool ${tool === 'pen' ? 'active' : ''}`}
              onClick={() => setTool('pen')}
            >
              ✏️ 筆（選擇）
            </button>
            <button
              className={`btn btn-tool ${tool === 'eraser' ? 'active' : ''}`}
              onClick={() => setTool('eraser')}
            >
              🧹 橡皮擦（取消）
            </button>
            <button className="btn btn-select" onClick={selectAll} disabled={faces.length === 0}>
              ☑️ 全選
            </button>
            <button className="btn btn-select" onClick={selectNone} disabled={faces.length === 0}>
              ✖️ 全不選
            </button>
          </div>

          <div className="image-container">
            {displayImage ? (
              <div className="image-wrapper">
                <img
                  src={displayImage}
                  alt="Preview"
                  onClick={handleImageClick}
                  style={{ cursor: faces.length > 0 && !processedImage ? 'pointer' : 'default' }}
                />
              </div>
            ) : (
              <div className="placeholder">請上傳圖片</div>
            )}
          </div>
        </section>

        <aside className="sidebar">
          <div className="panel">
            <h3>檢測結果</h3>
            <div className="result-list">
              {faces.length > 0 ? (
                <>
                  <p>檢測到 {faces.length} 個人臉（按面積從大到小排序）：</p>
                  {faces.map((face) => (
                    <div key={face.id} className={`face-item ${face.selected ? 'selected' : ''}`}>
                      #{face.id}: 面積={face.area}px², 置信度={face.confidence.toFixed(2)}
                      {face.selected ? ' ✓' : ''}
                    </div>
                  ))}
                </>
              ) : (
                <p>尚未檢測</p>
              )}
            </div>
          </div>

          <div className="panel">
            <h3>選擇狀態</h3>
            <p>已選擇 {selectedCount}/{faces.length} 個人臉進行遮蔽</p>
          </div>

          <div className="panel">
            <h3>遮蔽模式</h3>
            <div className="mode-selector">
              <button
                className={`btn btn-mode ${blurMode === 'emoji' ? 'active' : ''}`}
                onClick={() => {
                  setBlurMode('emoji');
                  if (faces.length > 0) fetchPreviewWithFaces(faces);
                }}
              >
                😊 表情符號
              </button>
              <button
                className={`btn btn-mode ${blurMode === 'blur' ? 'active' : ''}`}
                onClick={() => {
                  setBlurMode('blur');
                  if (faces.length > 0) fetchPreviewWithFaces(faces);
                }}
              >
                🌫️ 高斯模糊
              </button>
            </div>
          </div>

          <button
            className="btn btn-action btn-view"
            onClick={viewSelection}
            disabled={faces.length === 0}
          >
            👁️ 查看選擇
          </button>
          <button
            className="btn btn-action btn-blur"
            onClick={executeBlur}
            disabled={selectedCount === 0 || loading}
          >
            😊 執行遮蔽
          </button>
          <button
            className="btn btn-action btn-batch"
            onClick={() => batchInputRef.current?.click()}
            disabled={loading}
          >
            📦 批次遮蔽
          </button>
          <input
            ref={batchInputRef}
            type="file"
            accept="image/*"
            multiple
            onChange={handleBatchUpload}
            hidden
          />
          <button
            className="btn btn-action btn-save"
            onClick={saveResult}
            disabled={!processedImage}
          >
            💾 儲存結果
          </button>

          {batchProgress && (
            <div className="batch-progress">
              <p>處理中: {batchProgress.current}/{batchProgress.total}</p>
              <p className="filename">{batchProgress.filename}</p>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${(batchProgress.current / batchProgress.total) * 100}%` }}
                />
              </div>
            </div>
          )}
        </aside>
      </main>

      <footer className="footer">
        <span>☑️ {status || '就緒'}</span>
      </footer>
    </div>
  );
};

export default App;
