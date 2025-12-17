import React, { useState, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

// --- Configuration & Constants ---
const API_KEY = process.env.API_KEY;
const THEME = {
  primary: '#128C7E',
  secondary: '#25D366',
  bg: '#f0f2f5',
  card: '#ffffff',
  text: '#111b21',
  subtext: '#667781',
  userColors: ['#DCF8C6', '#E1F5FE', '#FFF9C4', '#F8BBD0', '#E0F2F1', '#D1C4E9', '#FFCCBC', '#B2DFDB', '#FFECB3']
};

// --- Types ---
interface Message {
  date: Date;
  author: string;
  content: string;
}

interface AnalysisStats {
  totalMessages: number;
  authorCounts: Record<string, number>;
  wordCounts: Record<string, number>;
  authorWordCounts: Record<string, Record<string, number>>; // New: Words per author
  emojiCounts: Record<string, number>; // New: Emojis
  hourlyActivity: Record<number, number>; 
  dailyActivity: Record<number, number>; 
  dailyVolume: Record<string, number>; // New: Timeline data (YYYY-MM-DD -> count)
  busiestDay: { date: string; count: number }; // New: Peak day
  participants: string[];
  dateRange: { start: Date; end: Date };
}

interface AIAnalysisResult {
  personalities: { author: string; description: string; sentiment: string }[];
  qa: { question: string; answer: string }[];
}

// --- Helpers ---
const STOP_WORDS = new Set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'un', 'por', 'con', 'no', 'una', 'su', 
'para', 'es', 'al', 'lo', 'como', 'más', 'o', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 
'todo', 'esta', 'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'muy', 'años', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 
'solo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 
'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'm', 'pm', 'am', 'omitted', 'image', 'audio', 'video', 'sticker', 'multimedia', 
'omitido', 'null', 'media']);

const DAYS = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];

// Emoji Regex (Broad matching for emoji characters)
const EMOJI_REGEX = /\p{Emoji_Presentation}/gu;

// --- Components ---

// 1. File Upload Component
const FileUpload = ({ onDataParsed }: { onDataParsed: (msgs: Message[]) => void }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const parseFile = (text: string) => {
    setError(null);
    const lines = text.split('\n');
    const messages: Message[] = [];
    
    // Regex Explained:
    // Support Android, iOS, and various locale formats
    const messageRegex = /^(?:\[?(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})[,.]?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[ap]\.?m\.?)?)(?:\]| -)?\s+)(.+?): (.+)/i;

    let currentMessage: Message | null = null;

    lines.forEach(line => {
      const cleanLine = line.replace(/[\u200e\u200f]/g, "").trim();
      if (!cleanLine) return;

      const match = cleanLine.match(messageRegex);

      if (match) {
        const [_, dateStr, timeStr, author, content] = match;

        // Parse Date
        let day, month, year;
        const parts = dateStr.split(/[-/.]/);
        
        if (parts[0].length === 4) { // YYYY-MM-DD
             year = parseInt(parts[0]);
             month = parseInt(parts[1]);
             day = parseInt(parts[2]);
        } else { // DD/MM/YYYY or MM/DD/YYYY
             // Defaulting to DD/MM/YYYY common in WhatsApp exports
             day = parseInt(parts[0]);
             month = parseInt(parts[1]);
             year = parseInt(parts[2]);
        }
        
        if (year < 100) year += 2000;

        // Parse Time
        let [hours, minutes] = timeStr.replace(/[ap]\.?m\.?/i, '').split(':').map(Number);
        if (timeStr.toLowerCase().includes('p') && hours < 12) hours += 12;
        if (timeStr.toLowerCase().includes('a') && hours === 12) hours = 0;

        // Ignore system messages if caught by regex (author usually distinct)
        if (author.includes("changed the subject") || author.includes("security code changed")) return;

        currentMessage = {
          date: new Date(year, month - 1, day, hours, minutes),
          author: author.trim(),
          content: content.trim()
        };
        messages.push(currentMessage);
      } else if (currentMessage) {
        currentMessage.content += `\n${cleanLine}`;
      }
    });

    if (messages.length === 0) {
        setError("No se pudieron detectar mensajes. Asegúrate de que es un archivo .txt exportado de WhatsApp.");
        return;
    }

    onDataParsed(messages);
  };

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text === 'string') parseFile(text);
    };
    reader.readAsText(file);
  };

  return (
    <div
      style={{
        border: `2px dashed ${error ? 'red' : (isDragging ? THEME.secondary : '#cbd5e0')}`,
        borderRadius: '1rem',
        padding: '3rem',
        textAlign: 'center',
        backgroundColor: isDragging ? '#e8f5e9' : 'white',
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        marginBottom: '2rem'
      }}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
      }}
    >
      <input
        type="file"
        accept=".txt"
        style={{ display: 'none' }}
        id="fileInput"
        onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
      />
      <label htmlFor="fileInput" style={{ cursor: 'pointer' }}>
        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📂</div>
        <h3 style={{ margin: '0 0 0.5rem', color: THEME.primary }}>Sube tu chat de WhatsApp (.txt)</h3>
        <p style={{ margin: 0, color: THEME.subtext }}>Arrastra el archivo o haz clic para buscar</p>
      </label>
      {error && (
          <div style={{ marginTop: '1rem', color: '#d32f2f', background: '#ffebee', padding: '0.5rem', borderRadius: '0.5rem' }}>
              ⚠️ {error}
          </div>
      )}
    </div>
  );
};

// 2. Timeline Chart (Messages per Day)
const TimelineChart = ({ dailyVolume, busiestDay }: { dailyVolume: Record<string, number>, busiestDay: { date: string, count: number } }) => {
    const data = Object.entries(dailyVolume).sort((a, b) => a[0].localeCompare(b[0]));
    if (data.length < 2) return null;

    const maxVal = busiestDay.count;
    const height = 150;
    const width = 100; // Percent

    // Simple SVG Polyline construction
    const points = data.map((d, i) => {
        const x = (i / (data.length - 1)) * 1000; // Map to 0-1000 coordinate space
        const y = height - ((d[1] / maxVal) * height);
        return `${x},${y}`;
    }).join(' ');

    return (
        <div style={{ width: '100%' }}>
            <div style={{ marginBottom: '1rem', fontSize: '0.9rem', color: THEME.subtext }}>
                🚀 Día más activo: <strong style={{ color: THEME.primary }}>{new Date(busiestDay.date).toLocaleDateString()}</strong> con <strong>{busiestDay.count}</strong> mensajes.
            </div>
            <svg viewBox={`0 0 1000 ${height}`} style={{ width: '100%', height: '150px', overflow: 'visible' }}>
                <polyline 
                    points={points} 
                    fill="none" 
                    stroke={THEME.primary} 
                    strokeWidth="2" 
                    vectorEffect="non-scaling-stroke"
                />
                <polygon 
                    points={`${points} 1000,${height} 0,${height}`} 
                    fill={THEME.primary} 
                    opacity="0.1" 
                />
            </svg>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#999', marginTop: '0.5rem' }}>
                <span>{new Date(data[0][0]).toLocaleDateString()}</span>
                <span>{new Date(data[data.length - 1][0]).toLocaleDateString()}</span>
            </div>
        </div>
    );
};

// 3. Simple Bar Chart (Reusable)
const SimpleBarChart = ({ data, color, labels, max }: { data: number[], labels: string[], color: string, max?: number }) => {
    const maxValue = max || Math.max(...data, 1);
    
    return (
        <div style={{ display: 'flex', alignItems: 'flex-end', height: '150px', gap: '4px' }}>
            {data.map((val, idx) => (
                <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', group: 'bar' } as any}>
                    <div style={{ 
                        width: '100%', 
                        background: color, 
                        opacity: 0.8,
                        height: `${(val / maxValue) * 100}%`,
                        minHeight: '2px',
                        borderRadius: '4px 4px 0 0',
                        transition: 'height 0.5s'
                    }} title={`${labels[idx]}: ${val}`}></div>
                    {labels.length <= 12 && ( // Only show labels if few items to avoid clutter
                        <span style={{ fontSize: '0.65rem', color: '#666', marginTop: '4px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', width: '100%', textAlign: 'center' }}>
                            {labels[idx]}
                        </span>
                    )}
                </div>
            ))}
        </div>
    );
};

// 4. Horizontal Bar Chart (For Authors & Emojis)
const HorizontalBarList = ({ items, colorPalette }: { items: { label: string, value: number, subLabel?: string }[], colorPalette?: string[] }) => {
    const maxVal = Math.max(...items.map(i => i.value), 1);
    
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            {items.map((item, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.9rem' }}>
                    <div style={{ width: '100px', textAlign: 'right', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontWeight: '500', color: '#444' }} title={item.label}>
                        {item.label}
                    </div>
                    <div style={{ flex: 1, background: '#f0f0f0', borderRadius: '4px', height: '1.25rem', overflow: 'hidden' }}>
                        <div style={{ 
                            width: `${(item.value / maxVal) * 100}%`, 
                            background: colorPalette ? colorPalette[idx % colorPalette.length] : THEME.secondary, 
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            paddingLeft: '0.5rem',
                            fontSize: '0.75rem',
                            color: '#333',
                            fontWeight: '600'
                        }}>
                            {item.value.toLocaleString()}
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
};

// 5. Per-Author Stats (Cards)
const AuthorWordCards = ({ authorWordCounts, colorPalette }: { authorWordCounts: Record<string, Record<string, number>>, colorPalette: string[] }) => {
    return (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            {Object.entries(authorWordCounts).map(([author, words], idx) => {
                 const topWords = Object.entries(words).sort((a,b) => b[1] - a[1]).slice(0, 5);
                 return (
                     <div key={author} style={{ background: '#f8f9fa', borderRadius: '0.5rem', padding: '1rem', borderTop: `4px solid ${colorPalette[idx % colorPalette.length]}` }}>
                         <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: '#333' }}>{author}</h4>
                         <ul style={{ paddingLeft: '1.2rem', margin: 0, fontSize: '0.85rem', color: '#555' }}>
                             {topWords.map(([w, c]) => (
                                 <li key={w}>{w} <span style={{ opacity: 0.6 }}>({c})</span></li>
                             ))}
                         </ul>
                     </div>
                 );
            })}
        </div>
    );
};


// 6. AI Analysis Component
const AIInsights = ({ messages, participants }: { messages: Message[], participants: string[] }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AIAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    if (!process.env.API_KEY) {
      setError("API Key missing");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const sampleSize = 400;
      const step = Math.max(1, Math.floor(messages.length / sampleSize));
      let sampleText = "";
      for(let i=0; i<messages.length; i += step) {
          if (sampleText.length > 20000) break;
          const m = messages[i];
          sampleText += `[${m.date.toISOString()}] ${m.author}: ${m.content.substring(0, 100)}\n`;
      }

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `Chat Log:\n${sampleText}\n\nAnaliza este chat.`,
        config: {
          systemInstruction: "Eres un Data Scientist experto en comportamiento social. Analiza el chat de WhatsApp. Sé conciso y usa emojis.",
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              personalities: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    author: { type: Type.STRING },
                    description: { type: Type.STRING, description: "Personalidad (max 30 palabras)" },
                    sentiment: { type: Type.STRING, description: "Emoji + Palabra (ej: '😂 Gracioso')" }
                  }
                }
              },
              qa: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    question: { type: Type.STRING },
                    answer: { type: Type.STRING }
                  }
                },
                description: "Genera 3 insights: 1. Dinámica de poder (quién lidera), 2. Tono emocional general, 3. Tema de conversación principal."
              }
            }
          }
        }
      });

      const jsonText = response.text;
      if (jsonText) {
        setResult(JSON.parse(jsonText));
      }
    } catch (err) {
      console.error(err);
      setError("Error al conectar con Gemini AI.");
    } finally {
      setLoading(false);
    }
  };

  if (!result && !loading) {
    return (
      <button 
        onClick={runAnalysis}
        style={{
          background: `linear-gradient(135deg, ${THEME.primary}, ${THEME.secondary})`,
          color: 'white',
          border: 'none',
          padding: '1rem 2rem',
          borderRadius: '2rem',
          fontSize: '1.1rem',
          fontWeight: 'bold',
          cursor: 'pointer',
          width: '100%',
          boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
        }}
      >
        ✨ Generar Análisis Psicológico (Gemini)
      </button>
    );
  }

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '2rem', color: THEME.subtext }}>
        <div className="spinner" style={{ marginBottom: '1rem', fontSize: '2rem', animation: 'spin 1s infinite linear' }}>🧠</div>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        <p>Procesando conversación...</p>
      </div>
    );
  }

  if (error) return <div style={{ color: 'red', textAlign: 'center' }}>{error}</div>;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
      <div style={{ gridColumn: '1 / -1', background: '#fff', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
        <h3 style={{ color: THEME.primary, marginTop: 0 }}>📊 Insights IA</h3>
        <div style={{ display: 'grid', gap: '1rem' }}>
            {result?.qa.map((item, idx) => (
                <div key={idx} style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '0.5rem', borderLeft: `4px solid ${THEME.primary}` }}>
                    <strong style={{ display: 'block', marginBottom: '0.5rem', color: '#333' }}>{item.question}</strong>
                    <span style={{ color: '#555' }}>{item.answer}</span>
                </div>
            ))}
        </div>
      </div>
      {result?.personalities.map((p, idx) => (
        <div key={idx} style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)', borderTop: `4px solid ${THEME.userColors[idx % THEME.userColors.length]}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
             <h3 style={{ margin: 0, color: '#333' }}>{p.author}</h3>
             <span style={{ background: '#e8f5e9', color: THEME.primary, padding: '0.25rem 0.5rem', borderRadius: '8px', fontSize: '0.9rem' }}>{p.sentiment}</span>
          </div>
          <p style={{ color: '#555', lineHeight: '1.6', fontSize: '0.95rem' }}>{p.description}</p>
        </div>
      ))}
    </div>
  );
};


// --- Main App Component ---

const App = () => {
  const [messages, setMessages] = useState<Message[] | null>(null);
  const [stats, setStats] = useState<AnalysisStats | null>(null);

  const processData = (msgs: Message[]) => {
    setMessages(msgs);

    const authorCounts: Record<string, number> = {};
    const wordCounts: Record<string, number> = {};
    const authorWordCounts: Record<string, Record<string, number>> = {};
    const emojiCounts: Record<string, number> = {};
    const hourlyActivity: Record<number, number> = {};
    const dailyActivity: Record<number, number> = {};
    const dailyVolume: Record<string, number> = {};

    let startDate = msgs[0].date;
    let endDate = msgs[msgs.length - 1].date;

    msgs.forEach(m => {
      // 1. Author Stats
      authorCounts[m.author] = (authorCounts[m.author] || 0) + 1;
      if (!authorWordCounts[m.author]) authorWordCounts[m.author] = {};

      // 2. Temporal Stats
      const hour = m.date.getHours();
      hourlyActivity[hour] = (hourlyActivity[hour] || 0) + 1;
      const day = m.date.getDay(); 
      dailyActivity[day] = (dailyActivity[day] || 0) + 1;
      const dateKey = m.date.toISOString().split('T')[0];
      dailyVolume[dateKey] = (dailyVolume[dateKey] || 0) + 1;

      // 3. Content Analysis
      // Emoji Count
      const emojis = m.content.match(EMOJI_REGEX) || [];
      emojis.forEach(e => {
          emojiCounts[e] = (emojiCounts[e] || 0) + 1;
      });

      // Word Count (Global & Per Author)
      const words = m.content.toLowerCase().split(/[\s,.;:!?()"]+/);
      words.forEach(w => {
        const cleanWord = w.trim();
        if (cleanWord.length > 3 && !STOP_WORDS.has(cleanWord) && isNaN(Number(cleanWord))) {
          // Global
          wordCounts[cleanWord] = (wordCounts[cleanWord] || 0) + 1;
          // Per Author
          authorWordCounts[m.author][cleanWord] = (authorWordCounts[m.author][cleanWord] || 0) + 1;
        }
      });
    });

    // Find Busiest Day
    let maxDayCount = 0;
    let busiestDayDate = "";
    Object.entries(dailyVolume).forEach(([date, count]) => {
        if (count > maxDayCount) {
            maxDayCount = count;
            busiestDayDate = date;
        }
    });

    setStats({
      totalMessages: msgs.length,
      authorCounts,
      wordCounts,
      authorWordCounts,
      emojiCounts,
      hourlyActivity,
      dailyActivity,
      dailyVolume,
      busiestDay: { date: busiestDayDate, count: maxDayCount },
      participants: Object.keys(authorCounts),
      dateRange: { start: startDate, end: endDate }
    });
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem', fontFamily: 'sans-serif' }}>
      <header style={{ marginBottom: '3rem', textAlign: 'center' }}>
        <h1 style={{ fontSize: '2.5rem', color: THEME.primary, marginBottom: '0.5rem', letterSpacing: '-1px' }}>
          WhatsApp Insights 💬
        </h1>
        <p style={{ color: THEME.subtext, fontSize: '1.1rem' }}>
          Descubre quién habla más, cuándo y qué dicen realmente.
        </p>
      </header>

      {!messages ? (
        <FileUpload onDataParsed={processData} />
      ) : (
        <div style={{ display: 'grid', gap: '2rem' }}>
          
          {/* Top Bar Info */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'white', padding: '1rem', borderRadius: '1rem', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
             <button 
                onClick={() => { setMessages(null); setStats(null); }}
                style={{ background: 'none', border: 'none', color: THEME.primary, cursor: 'pointer', fontSize: '1rem', fontWeight: 'bold' }}
             >
                ← Subir otro archivo
             </button>
             <div style={{ color: THEME.subtext, fontWeight: '500' }}>
                 📅 {stats?.dateRange.start.toLocaleDateString()} - {stats?.dateRange.end.toLocaleDateString()} 
                 <span style={{ margin: '0 10px' }}>|</span>
                 {stats?.totalMessages.toLocaleString()} mensajes
             </div>
          </div>

          {/* 1. Timeline Chart (Busiest Day) */}
          <div style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
             <h3 style={{ color: THEME.primary, marginTop: 0 }}>📅 Actividad Histórica</h3>
             {stats && <TimelineChart dailyVolume={stats.dailyVolume} busiestDay={stats.busiestDay} />}
          </div>

          {/* 2. Main Stats Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem' }}>
             
             {/* Who writes the most? */}
             <div style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
                <h3 style={{ color: THEME.primary, marginTop: 0 }}>🏆 ¿Quién escribe más?</h3>
                {stats && <HorizontalBarList 
                    items={(Object.entries(stats.authorCounts) as [string, number][]).sort(([,a], [,b]) => b - a).map(([k,v]) => ({ label: k, value: v }))} 
                    colorPalette={THEME.userColors} 
                />}
             </div>

             {/* Top Emojis */}
             <div style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
                <h3 style={{ color: THEME.primary, marginTop: 0 }}>😂 Emojis más usados</h3>
                {stats && Object.keys(stats.emojiCounts).length > 0 ? (
                    <HorizontalBarList 
                        items={(Object.entries(stats.emojiCounts) as [string, number][]).sort(([,a], [,b]) => b - a).slice(0, 8).map(([k,v]) => ({ label: k, value: v }))}
                        colorPalette={[THEME.secondary]} 
                    />
                ) : <div style={{ color: '#999' }}>No se encontraron emojis.</div>}
             </div>

             {/* Time Charts */}
             <div style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', gridColumn: 'span 1' }}>
                <h3 style={{ color: THEME.primary, marginTop: 0 }}>⏰ Actividad por Hora</h3>
                {stats && <SimpleBarChart 
                    data={Array.from({length: 24}, (_, i) => stats.hourlyActivity[i] || 0)} 
                    labels={['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']}
                    color={THEME.secondary}
                />}
             </div>

             <div style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', gridColumn: 'span 1' }}>
                <h3 style={{ color: THEME.primary, marginTop: 0 }}>🗓️ Días más activos</h3>
                {stats && (
                    <SimpleBarChart 
                        data={DAYS.map((_, i) => stats.dailyActivity[i] || 0)}
                        labels={DAYS}
                        color={THEME.primary}
                    />
                )}
             </div>
          </div>

          {/* 3. Words per Person */}
          <div style={{ background: 'white', padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}>
             <h3 style={{ color: THEME.primary, marginTop: 0 }}>🗣️ Palabras favoritas por persona</h3>
             {stats && <AuthorWordCards authorWordCounts={stats.authorWordCounts} colorPalette={THEME.userColors} />}
          </div>

          {/* 4. AI Section */}
          <div style={{ marginTop: '1rem' }}>
            <h2 style={{ color: '#333', textAlign: 'center', marginBottom: '1.5rem' }}>🧠 Análisis Profundo (Gemini)</h2>
            {stats && <AIInsights messages={messages} participants={stats.participants} />}
          </div>

        </div>
      )}
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);