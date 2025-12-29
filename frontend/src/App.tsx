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
  const [darkMode, setDarkMode] = useState(true);
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

  // 取得預覽圖片（帶人臉框）
  const fetchPreview = async (currentFaces: Face[]) => {
    if (!originalFile) return;

    const selectedIds = currentFaces.filter(f => f.selected).map(f => f.id);

    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('selected_ids', JSON.stringify(selectedIds));

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

      // 取得帶框的預覽圖
      await fetchPreviewWithFaces(detectedFaces);
    } catch (err) {
      setStatus('檢測失敗');
    } finally {
      setLoading(false);
    }
  };

  const fetchPreviewWithFaces = async (currentFaces: Face[]) => {
    if (!originalFile) return;

    const selectedIds = currentFaces.filter(f => f.selected).map(f => f.id);

    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('selected_ids', JSON.stringify(selectedIds));

    try {
      const res = await axios.post('/api/preview', formData, { responseType: 'blob' });
      setPreviewImage(URL.createObjectURL(res.data));
    } catch (err) {
      console.error('Preview failed:', err);
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

    // 更新預覽圖
    await fetchPreviewWithFaces(newFaces);
  };

  const selectAll = async () => {
    const newFaces = faces.map(f => ({ ...f, selected: true }));
    setFaces(newFaces);
    await fetchPreviewWithFaces(newFaces);
  };

  const selectNone = async () => {
    const newFaces = faces.map(f => ({ ...f, selected: false }));
    setFaces(newFaces);
    await fetchPreviewWithFaces(newFaces);
  };

  const executeBlur = async () => {
    if (!originalFile || faces.length === 0) return;
    setLoading(true);
    setStatus('處理中...');

    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('faces', JSON.stringify(faces.filter(f => f.selected)));

    try {
      const res = await axios.post('/api/blur', formData, { responseType: 'blob' });
      setProcessedImage(URL.createObjectURL(res.data));
      setPreviewImage(null);
      setStatus('遮蔽完成');
    } catch (err) {
      setStatus('處理失敗');
    } finally {
      setLoading(false);
    }
  };

  const saveResult = () => {
    if (processedImage) {
      const link = document.createElement('a');
      link.href = processedImage;
      link.download = `blurred_${originalFile?.name || 'image.jpg'}`;
      link.click();
    }
  };

  // 點擊圖片時的處理（需要計算點擊位置對應哪個人臉）
  const handleImageClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!faces.length || processedImage) return;

    const img = e.currentTarget;
    const rect = img.getBoundingClientRect();

    // 計算點擊在圖片上的相對位置（0-1）
    const relX = (e.clientX - rect.left) / rect.width;
    const relY = (e.clientY - rect.top) / rect.height;

    // 需要知道原圖尺寸來計算實際座標
    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    const clickX = relX * naturalWidth;
    const clickY = relY * naturalHeight;

    // 找到被點擊的人臉
    for (const face of faces) {
      if (clickX >= face.x1 && clickX <= face.x2 &&
          clickY >= face.y1 && clickY <= face.y2) {
        await toggleFace(face.id);
        break;
      }
    }
  };

  const selectedCount = faces.filter(f => f.selected).length;

  // 決定顯示哪張圖片
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
                  {faces.map((face, i) => (
                    <div key={face.id} className="face-item">
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
          >
            📦 批次遮蔽
          </button>
          <input ref={batchInputRef} type="file" accept="image/*" multiple hidden />
          <button
            className="btn btn-action btn-save"
            onClick={saveResult}
            disabled={!processedImage}
          >
            💾 儲存結果
          </button>
        </aside>
      </main>

      <footer className="footer">
        <span>☑️ {status || '就緒'}</span>
      </footer>
    </div>
  );
};

export default App;
