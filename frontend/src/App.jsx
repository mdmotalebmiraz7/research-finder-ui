import { useEffect, useRef, useState } from 'react';
import './App.css';

const API = '/api';
const EXAMPLE = 'AI-based fault detection in electrical power systems';
function cleanUrl(value) { if (!value) return ''; const m = value.match(/^\[[^\]]+\]\((https?:\/\/[^)]+)\)$/); return m ? m[1] : value; }

export default function App() {
  const [page, setPage] = useState('Discover');
  const [idea, setIdea] = useState(EXAMPLE);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('research-finder-groq-key') || '');
  const [keyDraft, setKeyDraft] = useState(() => localStorage.getItem('research-finder-groq-key') || '');
  const [showKeyModal, setShowKeyModal] = useState(() => !localStorage.getItem('research-finder-groq-key'));
  const [keyError, setKeyError] = useState('');
  const [status, setStatus] = useState('Checking…');
  const [loading, setLoading] = useState(false);
  const [liveLogs, setLiveLogs] = useState([]);
  const logOffset = useRef(0);
  const [error, setError] = useState('');
  const [research, setResearch] = useState(null);
  const [papers, setPapers] = useState([]);
  const [selected, setSelected] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [processed, setProcessed] = useState(null);
  const [question, setQuestion] = useState('');
  const [chat, setChat] = useState([]);
  const [asking, setAsking] = useState(false);
  const [saved, setSaved] = useState([]);

  useEffect(() => { fetch(`${API}/health`).then(r => { if (!r.ok) throw new Error(); setStatus('Connected'); }).catch(() => setStatus('Not connected')); }, []);

  function normalizeLogs(payload) {
    const rows = Array.isArray(payload) ? payload : (payload?.logs || payload?.items || payload?.messages || []);
    if (!Array.isArray(rows)) return [];
    return rows.map((item) => typeof item === 'string' ? item : (item?.message || item?.text || item?.log || JSON.stringify(item)));
  }

  async function readBackendLogs() {
    const res = await fetch(`${API}/logs`, { cache: 'no-store' });
    if (!res.ok) throw new Error('Could not load backend logs');
    return normalizeLogs(await res.json());
  }

  // Poll the backend log endpoint while research is running so messages appear as they happen.
  useEffect(() => {
    if (!loading) return undefined;
    let active = true;
    const poll = async () => {
      try {
        const allLogs = await readBackendLogs();
        if (active) setLiveLogs(allLogs.slice(logOffset.current));
      } catch (_) {
        // Research itself can still succeed if log polling is temporarily unavailable.
      }
    };
    poll();
    const timer = setInterval(poll, 900);
    return () => { active = false; clearInterval(timer); };
  }, [loading]);

  async function searchPapers(e) {
    e?.preventDefault(); setError('');
    if (!idea.trim()) return setError('Enter a research topic first.');
    if (!apiKey.trim()) { setShowKeyModal(true); return setError('Add your Groq API key to start a research search.'); }
    setResearch(null); setPapers([]); setLiveLogs([]);
    // Snapshot existing logs so previous searches are not mixed into this run.
    try { logOffset.current = (await readBackendLogs()).length; } catch (_) { logOffset.current = 0; }
    setLoading(true);
    try {
      const res = await fetch(`${API}/research`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ idea: idea.trim(), api_key: apiKey.trim() }) });
      const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'Research search failed.');
      setResearch(data); setPapers(data.papers || []);
      try {
        const allLogs = await readBackendLogs();
        setLiveLogs(allLogs.slice(logOffset.current));
      } catch (_) {}
    } catch (err) { setError(err.message || 'Could not search papers.'); }
    finally { setLoading(false); }
  }

  async function openPaper(paper) {
    setSelected(paper); setPage('Reader'); setError(''); setChat([]); setProcessed(null);
    setProcessing(true);
    try {
      let res = await fetch(`${API}/paper`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rank: paper.Rank }) });
      let data = await res.json(); if (!res.ok) throw new Error(data.detail || 'Could not select paper.');
      res = await fetch(`${API}/paper/process`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rank: paper.Rank }) });
      data = await res.json(); if (!res.ok) throw new Error(data.detail || 'PDF processing failed.');
      setProcessed(data);
    } catch (err) { setError(err.message || 'Could not process PDF. You can still open the PDF link.'); }
    finally { setProcessing(false); }
  }

  async function ask(e) {
    e.preventDefault(); if (!question.trim() || asking) return;
    if (!apiKey.trim()) { setShowKeyModal(true); return setError('Add your Groq API key before asking a question.'); }
    if (!processed) return setError('Wait until PDF processing finishes.');
    const q = question.trim(); setQuestion(''); setError(''); setChat(old => [...old, { role: 'user', text: q }]); setAsking(true);
    try {
      const res = await fetch(`${API}/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q, api_key: apiKey.trim() }) }); const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'Question failed.'); const answer = data.answer;
      setChat(old => [...old, { role: 'assistant', text: answer }]);
    } catch (err) { setChat(old => [...old, { role: 'assistant', text: err.message || 'Could not answer.', failed: true }]); }
    finally { setAsking(false); }
  }

  function saveApiKey(e) {
    e.preventDefault();
    const value = keyDraft.trim();
    if (!value) { setKeyError('Please enter your Groq API key.'); return; }
    localStorage.setItem('research-finder-groq-key', value);
    setApiKey(value);
    setKeyDraft(value);
    setKeyError('');
    setShowKeyModal(false);
  }

  function toggleSave(paper) { setSaved(old => old.some(p => p.title === paper.title) ? old.filter(p => p.title !== paper.title) : [...old, paper]); }

  return <div className="app">
    <header className="topbar"><button className="brand" onClick={() => setPage('Discover')}><span className="brand-icon">R</span><span className="brand-copy">Research Finder<small>by Md Miraz Ali</small></span></button><nav><button className={page === 'Discover' ? 'nav active' : 'nav'} onClick={() => setPage('Discover')}>Discover</button><button className={page === 'Library' ? 'nav active' : 'nav'} onClick={() => setPage('Library')}>Saved papers <span className="count">{saved.length}</span></button></nav><button className="key-settings" onClick={() => { setKeyDraft(apiKey); setKeyError(''); setShowKeyModal(true); }}>API key</button><div className="status"><i className={status === 'Connected' ? 'dot good' : 'dot'} />{status}</div></header>
    {error && <div className="error"><span>{error}</span><button onClick={() => setError('')}>×</button></div>}

    {page === 'Discover' && <main className="content">
      <div className="intro"><p className="eyebrow">RESEARCH WORKSPACE</p><h1>Find relevant research papers.</h1><p>Search arXiv, rank papers, then read and ask questions using RAG.</p></div>
      <form className="search-panel" onSubmit={searchPapers}><label htmlFor="topic">Research topic</label><textarea id="topic" rows="2" value={idea} onChange={e => setIdea(e.target.value)} placeholder="Describe your research idea..."/><div className="form-bottom"><p className="key-hint">Groq API key saved for this browser. <button type="button" onClick={() => { setKeyDraft(apiKey); setKeyError(''); setShowKeyModal(true); }}>Change key</button></p><div className="search-actions"><button className="primary" disabled={loading}>{loading ? 'Searching…' : 'Search papers'}</button></div></div></form>
      {(loading || liveLogs.length > 0) && <section className="live-log-panel" aria-live="polite">
        <div className="live-log-heading"><div><span className={loading ? 'live-indicator running' : 'live-indicator'} /> <strong>{loading ? 'Research in progress' : 'Research activity'}</strong></div><span className="live-log-count">{liveLogs.length} events</span></div>
        <div className="live-log-list">{liveLogs.length === 0 ? <div className="live-log-line muted">Connecting to backend logs…</div> : liveLogs.map((line, i) => <div className="live-log-line" key={`${i}-${line}`}><span className="log-bullet">›</span><span>{line.replace(/^\[INFO\]\s*/, '')}</span></div>)}{loading && <div className="live-log-line muted"><span className="log-bullet pulse">›</span><span>Waiting for next backend event…</span></div>}</div>
      </section>}
      {research && <section className="summary"><h2>Research summary</h2><p>{research.summary}</p>{research.titles?.length > 0 && <div className="topics"><strong>Search topics</strong><div>{research.titles.map((t,i)=><span key={i}>{t}</span>)}</div></div>}</section>}
      <div className="results-heading"><div><h2>Ranked papers</h2><p>{papers.length ? `${papers.length} papers found` : 'Your search results will appear here.'}</p></div></div>
      <div className="paper-list">{papers.map((paper,i)=><article className="paper-card" key={`${paper.Rank}-${paper.title}`}><div className="rank">{paper.Rank || i+1}</div><div className="paper-info"><h3>{paper.title}</h3><div className="paper-meta"><span>{paper.year || 'Year unavailable'}</span><span>Relevance {typeof paper.Semantic === 'number' ? paper.Semantic.toFixed(2) : '—'}</span></div><p>{paper.abstract || 'No abstract available.'}</p><div className="paper-actions"><button className="primary small" onClick={() => openPaper(paper)}>Open reader</button><a href={cleanUrl(paper.url || paper.link)} target="_blank" rel="noreferrer">Abstract ↗</a><button className="save" onClick={() => toggleSave(paper)}>{saved.some(p => p.title === paper.title) ? 'Saved ✓' : 'Save paper'}</button></div></div></article>)}</div>
    </main>}

    {page === 'Reader' && <main className="reader-page"><div className="reader-heading"><button className="back" onClick={() => setPage('Discover')}>← Back to papers</button><div><h1>{selected?.title || 'Paper reader'}</h1><p>{processing ? 'Downloading PDF and preparing retrieval…' : processed ? `PDF ready · ${processed.total_pages} pages · ${processed.total_chunks} chunks` : 'PDF preview and RAG assistant'}</p></div><a className="external" href={cleanUrl(selected?.pdf_url)} target="_blank" rel="noreferrer">Open PDF ↗</a></div><div className="reader-layout"><section className="pdf-panel"><div className="panel-title">PDF READER</div>{selected?.pdf_url ? <iframe title="Research paper PDF" src={cleanUrl(selected.pdf_url)} /> : <div className="empty">Select a paper from Discover.</div>}</section><section className="chat-panel"><div className="panel-title">ASK THIS PAPER <span className={processed ? 'ready' : 'not-ready'}>{processing ? 'Preparing…' : processed ? 'Ready' : 'Not processed'}</span></div><div className="chat-messages">{!chat.length && <div className="chat-empty"><div className="chat-symbol">✦</div><h3>Ask about this paper</h3><p>Ask about its methods, findings, limitations, or key concepts.</p><div className="suggestions"><button onClick={() => setQuestion('Summarize the main findings.')}>Summarize the main findings</button><button onClick={() => setQuestion('What method does this paper propose?')}>Explain the proposed method</button><button onClick={() => setQuestion('What are the limitations?')}>What are its limitations?</button></div></div>}{chat.map((m,i)=><div className={`message ${m.role} ${m.failed ? 'failed' : ''}`} key={i}><span>{m.role === 'user' ? 'You' : 'RAG assistant'}</span><p>{m.text}</p></div>)}{asking && <div className="message assistant"><span>RAG assistant</span><p>Thinking…</p></div>}</div><form className="chat-form" onSubmit={ask}><textarea rows="2" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about this paper…"/><button className="primary" disabled={asking || !question.trim()}>{asking ? '…' : 'Send ↑'}</button></form><p className="privacy-note">Answers use the selected PDF's retrieved context.</p></section></div></main>}

    {page === 'Library' && <main className="content"><div className="intro"><p className="eyebrow">YOUR LIBRARY</p><h1>Saved papers.</h1><p>Keep track of papers you want to read later.</p></div>{!saved.length ? <div className="empty-library">No saved papers yet. Find a paper and click “Save paper”.</div> : <div className="paper-list">{saved.map((paper,i)=><article className="paper-card" key={paper.title}><div className="rank">{i+1}</div><div className="paper-info"><h3>{paper.title}</h3><div className="paper-meta"><span>{paper.year || 'Year unavailable'}</span></div><p>{paper.abstract || 'No abstract available.'}</p><div className="paper-actions"><button className="primary small" onClick={() => openPaper(paper)}>Open reader</button><button className="save" onClick={() => toggleSave(paper)}>Remove</button></div></div></article>)}</div>}</main>}
    <footer><strong>Research Finder</strong> <span>·</span> AI-assisted literature discovery<div className="creator-credit">Created by <strong>Md Miraz Ali</strong> · Dhaka University · EEE Department</div></footer>
    {showKeyModal && <div className="modal-backdrop" role="presentation"><section className="key-modal" role="dialog" aria-modal="true" aria-labelledby="key-modal-title"><div className="modal-brand"><span className="brand-icon">R</span><span>Research Finder</span></div><p className="eyebrow">PERSONALIZE YOUR WORKSPACE</p><h2 id="key-modal-title">Connect your AI workspace.</h2><p className="modal-description">Enter your Groq API key once to search research papers and ask questions about PDFs. It will be saved in this browser only.</p><form onSubmit={saveApiKey}><label htmlFor="groq-key">Groq API key</label><input id="groq-key" type="password" autoComplete="off" autoFocus value={keyDraft} onChange={e => setKeyDraft(e.target.value)} placeholder="gsk_…"/><p className="key-help">Your key is stored in this browser's local storage and sent to your connected backend when you use AI features.</p>{keyError && <p className="key-error">{keyError}</p>}<button className="primary modal-submit" type="submit">Save key & continue <span>→</span></button></form><div className="modal-credit">Built by <strong>Md Miraz Ali</strong><br/>Dhaka University · EEE Department</div></section></div>}
  </div>;
}
