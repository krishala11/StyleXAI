"use client";

import { useState } from "react";
import styles from "./page.module.css";

const API_BASE = "http://localhost:8000";

type Profile = {
  gender: string;
  age: number;
};

export default function Home() {
  const [mode, setMode] = useState<"chat" | "guided">("chat");
  const [profile, setProfile] = useState<Profile>({
    gender: "men",
    age: 24,
  });

  const [query, setQuery] = useState("I need an outfit for a business meeting.");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  // Guided Mode State
  const [guidedStep, setGuidedStep] = useState(1);
  const [guidedSelections, setGuidedSelections] = useState<any>({});
  const [guidedData, setGuidedData] = useState<any>(null);

  const handleRecommend = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, profile }),
      });
      
      if (!res.ok) throw new Error("API Request Failed");
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend. Is uvicorn running?");
    } finally {
      setLoading(false);
    }
  };

  const fetchGuided = async (step: number, selections: any) => {
    setLoading(true);
    setError("");
    try {
      const payload = {
        query,
        profile,
        selections: { step, ...selections }
      };
      const res = await fetch(`${API_BASE}/api/guided`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error("API Request Failed");
      const data = await res.json();
      setGuidedData(data);
      setGuidedStep(step);
      
      // If backend says to skip bottom (e.g. for a one-piece dress), auto-advance to step 3
      if (data.skip_bottom && step === 2) {
        fetchGuided(3, selections);
      }
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleStartGuided = (e: React.FormEvent) => {
    e.preventDefault();
    setGuidedSelections({});
    fetchGuided(1, {});
  };

  const handleGuidedSelect = (slot: string, id: string) => {
    const newSelections = { ...guidedSelections, [slot]: id };
    setGuidedSelections(newSelections);
    fetchGuided(guidedStep + 1, newSelections);
  };

  const renderOutfitCard = (outfit: any, idx: number) => (
    <div key={idx} className={styles.outfitCard}>
      <div className={styles.outfitHeader}>
        <h3>Outfit {idx + 1}</h3>
        <span className={styles.outfitScore}>
          Match Score: {(outfit.scores.outfit_score * 100).toFixed(1)}%
        </span>
      </div>
      
      <div className={styles.itemsGrid}>
        {Object.entries(outfit.items).map(([slot, item]: [string, any]) => (
          <div key={slot} className={styles.itemCard}>
            <img 
              src={`${API_BASE}/data/${item.image}`} 
              alt={item.name} 
              className={styles.itemImage}
            />
            <div className={styles.itemSlot}>{slot}</div>
            <div className={styles.itemName}>{item.name.substring(0, 40)}</div>
          </div>
        ))}
      </div>

      {outfit.explanation && (
        <div className={styles.explanation}>
          💡 {outfit.explanation}
        </div>
      )}
    </div>
  );

  return (
    <main className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>StyleXAI</h1>
        <p className={styles.subtitle}>Multimodal Retrieval & Graph Compatibility Engine</p>
      </header>

      <div className={styles.tabs}>
        <button 
          className={`${styles.tabBtn} ${mode === "chat" ? styles.active : ""}`}
          onClick={() => setMode("chat")}
        >
          Chat Mode
        </button>
        <button 
          className={`${styles.tabBtn} ${mode === "guided" ? styles.active : ""}`}
          onClick={() => setMode("guided")}
        >
          Guided Builder
        </button>
      </div>

      <div className={styles.mainLayout}>
        <aside className={`${styles.sidebar} ${styles.glassPanel}`}>
          <h2>Your Profile</h2>
          <div className={styles.formGroup}>
            <label>Gender</label>
            <select 
              value={profile.gender} 
              onChange={e => setProfile({...profile, gender: e.target.value})}
            >
              <option value="men">Men</option>
              <option value="women">Women</option>
            </select>
          </div>
          <div className={styles.formGroup}>
            <label>Age</label>
            <input 
              type="number" 
              value={profile.age} 
              onChange={e => setProfile({...profile, age: parseInt(e.target.value)})}
            />
          </div>
        </aside>

        <section className={`${styles.contentArea} ${styles.glassPanel}`}>
          {mode === "chat" && (
            <>
              <form onSubmit={handleRecommend} className={styles.chatForm}>
                <input 
                  type="text" 
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  className={styles.chatInput}
                  placeholder="Describe what you want to wear..."
                />
                <button type="submit" className={`btn-primary ${styles.submitBtn}`} disabled={loading}>
                  {loading ? "Thinking..." : "Recommend"}
                </button>
              </form>

              {error && <div style={{color: "red", padding: "1rem"}}>{error}</div>}
              {loading && <div className={styles.loading}>Analyzing intent, matching semantics, and evaluating topological compatibility...</div>}

              {result && result.outfits && (
                <div className={`${styles.outfitsGrid} animate-fade-up`}>
                  {result.outfits.map(renderOutfitCard)}
                </div>
              )}
            </>
          )}

          {mode === "guided" && (
            <div className={`${styles.guidedContainer} animate-fade-up`}>
              <form onSubmit={handleStartGuided} className={styles.chatForm}>
                <input 
                  type="text" 
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  className={styles.chatInput}
                  placeholder="Describe your desired outfit..."
                />
                <button type="submit" className={`btn-primary ${styles.submitBtn}`} disabled={loading}>
                  {loading ? "Starting..." : "Start Builder"}
                </button>
              </form>

              {error && <div style={{color: "red", padding: "1rem"}}>{error}</div>}
              {loading && <div className={styles.loading}>Running multimodal retrieval & compatibility expansion...</div>}

              {guidedData && guidedStep < 5 && !loading && (
                <div className={styles.guidedProgress}>
                  <button 
                    className={styles.startOverBtn} 
                    onClick={() => { setGuidedData(null); setGuidedStep(1); }}
                  >
                    Start Over
                  </button>
                  <h3>Step {guidedStep}: {guidedData.message}</h3>
                  <div className={`${styles.candidatesGrid} animate-fade-up`}>
                    {guidedData.candidates?.map((c: any) => (
                      <div 
                        key={c.id} 
                        className={styles.candidateCard}
                        onClick={() => handleGuidedSelect(
                          guidedStep === 1 ? 'topwear' : 
                          guidedStep === 2 ? 'bottomwear' : 
                          guidedStep === 3 ? 'footwear' : 'accessory', 
                          c.id
                        )}
                      >
                        <img src={`${API_BASE}/data/${c.image}`} alt={c.name} />
                        <p>{c.name.substring(0, 30)}</p>
                        {c.pair_score && <span className={styles.badge} style={{top: '12px', left: '12px', right: 'auto', background: '#3b82f6', border: 'none', color: '#ffffff', boxShadow: '0 4px 6px rgba(59,130,246,0.3)'}}>Match: {(c.pair_score * 100).toFixed(0)}%</span>}
                        {c.is_strict_match === false && <span className={styles.badge} style={{backgroundColor: '#f59e0b', color: '#ffffff', border: 'none', boxShadow: '0 4px 6px rgba(245,158,11,0.3)'}}>Creative Match</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {guidedData && guidedStep === 5 && guidedData.outfits && !loading && (
                <div>
                  <button 
                    className={styles.startOverBtn} 
                    onClick={() => { setGuidedData(null); setGuidedStep(1); }}
                    style={{ marginBottom: '2rem' }}
                  >
                    Start Over
                  </button>
                  <h3 style={{ marginBottom: '1.5rem', textAlign: 'center', color: 'var(--primary)' }}>
                    Your Custom Outfit is Ready!
                  </h3>
                  <div className={styles.outfitsGrid}>
                    {guidedData.outfits.map(renderOutfitCard)}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
