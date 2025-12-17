import React, { useState, useMemo, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

// --- 1. CONFIG & STYLES (Separation of Concerns) ---
const THEME = {
  primary: '#128C7E',
  secondary: '#25D366',
  bg: '#eef1f4',
  card: '#ffffff',
  text: '#111b21',
  subtext: '#667781',
  danger: '#ef5350',
  userColors: ['#DCF8C6', '#E1F5FE', '#FFF9C4', '#F8BBD0', '#E0F2F1', '#D1C4E9', '#FFCCBC', '#B2DFDB', '#FFECB3']
};

// Estilos extraídos para limpiar el JSX (CSS-in-JS "lite")
const styles = {
  container: { maxWidth: '1200px', margin: '0 auto', padding: '2rem', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif' },
  card: { background: THEME.card, padding: '1.5rem', borderRadius: '1rem', boxShadow: '0 4px 12px rgba(0,0,0,0.05)', transition: 'transform 0.2s' },
  header: { textAlign: 'center' as const, marginBottom: '3rem' },
  grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' },
  flexCenter: { display: 'flex', justifyContent: 'center', alignItems: 'center' },
  uploadBox: (isDragging: boolean, error: boolean) => ({
    border: `2px dashed ${error ? THEME.danger : (isDragging ? THEME.secondary : '#cbd5e0')}`,
    borderRadius: '1rem', padding: '3rem', textAlign: 'center' as const,
    backgroundColor: isDragging ? '#e8f5e9' : 'white', cursor: 'pointer', transition: 'all 0.3s ease'
  })
};

// --- 2. TYPES ---
interface Message {
  date: Date;
  author: string;
  content: string;
}

interface AnalysisStats {
  totalMessages: number;
  authorCounts: Record<string, number>;
  wordCounts: Record<string, number>;
  authorWordCounts: Record<string, Record<string, number>>;
  emojiCounts: Record<string, number>;
  hourlyActivity: Record<number, number>;
  dailyActivity: Record<number, number>; // 0-6 (Domingo-Sabado)
  dailyVolume: Record<string, number>;
  busiestDay: { date: string; count: number };
  participants: string[];
  dateRange: { start: Date; end: Date };
}

interface AIAnalysisResult {
  personalities: { author: string; description: string; sentiment: string }[];
  qa: { question: string; answer: string }[];
}

// --- 3. UTILS (Pure Functions outside component) ---
const STOP_WORDS = new Set(['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'se', 'del', 'las', 'un', 'por', 'con', 'no', 'una', 'su', 'para', 'es', 'al', 'lo', 'como', 'más', 'o', 'pero', 'sus', 'le', 'ha', 'me', 'si', 'sin', 'sobre', 'este', 'ya', 'entre', 'cuando', 'todo', 'esta', 'ser', 'son', 'dos', 'también', 'fue', 'había', 'era', 'muy', 'años', 'hasta', 'desde', 'está', 'mi', 'porque', 'qué', 'solo', 'han', 'yo', 'hay', 'vez', 'puede', 'todos', 'así', 'nos', 'ni', 'parte', 'tiene', 'él', 'uno', 'donde', 'bien', 'tiempo', 'mismo', 'ese', 'ahora', 'cada', 'e', 'vida', 'otro', 'después', 'te', 'm', 'pm', 'am', 'omitted', 'image', 'audio', 'video', 'sticker', 'multimedia', 'omitido', 'null', 'media']);
const DAYS = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];
const EMOJI_REGEX = /\p{Emoji_Presentation}/gu;

// Parsing optimizado

const parseChatFile = (text: string): Message[] => {
  const lines = text.split('\n');
  const messages: Message[] = [];
  
  // REGEX FLEXIBLE:
  // 1. Fecha (DD/MM/YYYY o YYYY-MM-DD)
  // 2. Hora (captura todo hasta el guión separador)
  // 3. Separador (puede ser " - " o "] ")
  // 4. Autor (todo hasta los dos puntos)
  const messageRegex = /^(?:\[?(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})[,.]?\s+)(.*?)(?:\s+[-]\s+|\]\s+)([^:]+): (.+)/i;

  let currentMessage: Message | null = null;

  for (const line of lines) {
    if (line.length < 5) continue;
    
    // --- 1. SANITIZACIÓN (Clave para arreglar tu bug) ---
    // Reemplazamos espacios invisibles (\u202f, \u00a0) por espacios normales.
    // Esto arregla el bug de "p. m." que tiene un espacio raro en medio.
    const cleanLine = line
        .replace(/[\u200e\u200f]/g, "") // Caracteres de dirección de texto
        .replace(/[\u202f\u00a0]/g, " ") // Espacios duros/estrechos -> espacio normal
        .trim();

    const match = cleanLine.match(messageRegex);

    if (match) {
      let [_, dateStr, rawTimeStr, author, content] = match;

      // Filtros de mensajes de sistema
      if (author.includes("changed the subject") || author.includes("security code") || author.includes("cambió el asunto")) continue;

      // --- 2. PARSING DE FECHA ---
      const dateParts = dateStr.split(/[-/.]/).map(Number);
      let day, month, year;
      if (dateParts[0] > 31 || dateParts[0] > 1000) { // Formato YYYY-MM-DD
         [year, month, day] = dateParts;
      } else { // Formato DD-MM-YYYY
         [day, month, year] = dateParts;
      }
      if (year < 100) year += 2000;

      // --- 3. PARSING DE HORA ROBUSTO (12h/24h) ---
      const timeStrLower = rawTimeStr.toLowerCase().trim();
      const timeNumbers = timeStrLower.match(/(\d{1,2}):(\d{2})/);
      
      let hours = 0, minutes = 0;
      if (timeNumbers) {
          [hours, minutes] = [parseInt(timeNumbers[1]), parseInt(timeNumbers[2])];
          
          // Detección de PM/AM buscando 'p' o 'a' en cualquier parte de la cadena de hora
          const hasP = /[bp]/.test(timeStrLower.replace(/[\d:]/g, '')); // Buscamos p, pm, p.m.
          const isPM = timeStrLower.includes('p'); 
          
          if (hasP) {
              if (isPM && hours < 12) hours += 12;
              if (!isPM && hours === 12) hours = 0;
          }
      }

      // --- 4. RED DE SEGURIDAD PARA EL AUTOR ---
      // Si por alguna razón el regex falló y el autor sigue siendo "p. m. - Rebeca",
      // esto lo detecta y lo corta manualmente.
      author = author.trim();
      if (author.match(/^[ap]\.?\s?m\.?\s*-\s*/i)) {
          // Si el autor empieza con "p. m. - ", cortamos por el guión y nos quedamos con la parte derecha
          const parts = author.split('-');
          if (parts.length > 1) {
              author = parts[parts.length - 1].trim();
          }
      }

      currentMessage = {
        date: new Date(year, month - 1, day, hours, minutes),
        author: author,
        content: content.trim()
      };
      messages.push(currentMessage);
    } else if (currentMessage) {
      // Mensajes multilinea
      currentMessage.content += `\n${cleanLine}`;
    }
  }
  return messages;
};

// Estadísticas optimizadas (Single Pass Loop)
const calculateStats = (msgs: Message[]): AnalysisStats => {
  const stats: AnalysisStats = {
    totalMessages: msgs.length,
    authorCounts: {},
    wordCounts: {},
    authorWordCounts: {},
    emojiCounts: {},
    hourlyActivity: {},
    dailyActivity: {},
    dailyVolume: {},
    busiestDay: { date: '', count: 0 },
    participants: [],
    dateRange: { start: msgs[0].date, end: msgs[msgs.length - 1].date }
  };

  msgs.forEach(m => {
    // 1. Contadores básicos
    stats.authorCounts[m.author] = (stats.authorCounts[m.author] || 0) + 1;
    stats.hourlyActivity[m.date.getHours()] = (stats.hourlyActivity[m.date.getHours()] || 0) + 1;
    stats.dailyActivity[m.date.getDay()] = (stats.dailyActivity[m.date.getDay()] || 0) + 1;
    
    const dateKey = m.date.toISOString().split('T')[0];
    stats.dailyVolume[dateKey] = (stats.dailyVolume[dateKey] || 0) + 1;
    if (stats.dailyVolume[dateKey] > stats.busiestDay.count) {
      stats.busiestDay = { date: dateKey, count: stats.dailyVolume[dateKey] };
    }

    if (!stats.authorWordCounts[m.author]) stats.authorWordCounts[m.author] = {};

    // 2. Análisis de contenido (Emojis y Palabras)
    const emojis = m.content.match(EMOJI_REGEX);
    if (emojis) {
        for (const e of emojis) stats.emojiCounts[e] = (stats.emojiCounts[e] || 0) + 1;
    }

    // Tokenización simple pero efectiva
    const words = m.content.toLowerCase().split(/[\s,.;:!?()"]+/);
    for (const w of words) {
        if (w.length > 3 && !STOP_WORDS.has(w) && isNaN(Number(w))) {
            stats.wordCounts[w] = (stats.wordCounts[w] || 0) + 1;
            stats.authorWordCounts[m.author][w] = (stats.authorWordCounts[m.author][w] || 0) + 1;
        }
    }
  });

  stats.participants = Object.keys(stats.authorCounts);
  return stats;
};


// --- 4. COMPONENTS ---

const FileUpload = ({ onDataParsed }: { onDataParsed: (msgs: Message[]) => void }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const processFile = async (file: File) => {
    setLoading(true);
    setError(null);
    
    // Usamos setTimeout para permitir que React renderice el estado de "Cargando"
    // antes de bloquear el thread con el parsing.
    setTimeout(async () => {
        try {
            const text = await file.text();
            const messages = parseChatFile(text);
            if (messages.length === 0) throw new Error("No se encontraron mensajes válidos.");
            onDataParsed(messages);
        } catch (e) {
            setError("Error al leer el archivo. Asegúrate que es un .txt de WhatsApp.");
        } finally {
            setLoading(false);
        }
    }, 100);
  };

  return (
    <div
      style={styles.uploadBox(isDragging, !!error)}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault(); setIsDragging(false);
        if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
      }}
    >
      <input
        type="file" accept=".txt" style={{ display: 'none' }} id="fileInput"
        onChange={(e) => { if (e.target.files?.[0]) processFile(e.target.files[0]); }}
      />
      <label htmlFor="fileInput" style={{ cursor: 'pointer' }}>
        <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>{loading ? '⏳' : '📂'}</div>
        <h3 style={{ margin: '0 0 0.5rem', color: THEME.primary }}>
            {loading ? 'Analizando conversación...' : 'Sube tu chat de WhatsApp (.txt)'}
        </h3>
        <p style={{ margin: 0, color: THEME.subtext }}>
            {loading ? 'Esto puede tomar unos segundos.' : 'Arrastra el archivo o haz clic para buscar'}
        </p>
      </label>
      {error && <div style={{ marginTop: '1rem', color: THEME.danger }}>⚠️ {error}</div>}
    </div>
  );
};

const HorizontalBarList = ({ items, colorPalette }: { items: { label: string, value: number }[], colorPalette?: string[] }) => {
    const maxVal = Math.max(...items.map(i => i.value), 1);
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
            {items.map((item, idx) => (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.9rem' }}>
                    <div style={{ width: '100px', textAlign: 'right', fontWeight: '500', color: '#444', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.label}</div>
                    <div style={{ flex: 1, background: '#f0f0f0', borderRadius: '4px', height: '1.25rem' }}>
                        <div style={{ 
                            width: `${(item.value / maxVal) * 100}%`, 
                            background: colorPalette ? colorPalette[idx % colorPalette.length] : THEME.secondary, 
                            height: '100%', borderRadius: '4px', display: 'flex', alignItems: 'center', paddingLeft: '0.5rem', fontSize: '0.75rem', color: '#333', fontWeight: 'bold'
                        }}>
                            {item.value.toLocaleString()}
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );
};

const AIInsights = ({ messages, participants }: { messages: Message[], participants: string[] }) => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AIAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = async () => {
    if (!process.env.API_KEY) {
      setError("Falta la API Key. Configura API_KEY en tu entorno.");
      return;
    }
    setLoading(true); setError(null);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      
      // ESTRATEGIA MEJORADA: Sampling inteligente
      const chunkSize = 50;
      const chunks = [];
      
      // Bloque 1: Inicio
      chunks.push(...messages.slice(0, chunkSize));
      // Bloque 2: Mitad
      const mid = Math.floor(messages.length / 2);
      chunks.push(...messages.slice(mid, mid + chunkSize));
      // Bloque 3: Final
      chunks.push(...messages.slice(-chunkSize));

      const sampleText = chunks.map(m => `[${m.author}]: ${m.content}`).join('\n');

      const prompt = `
        Analiza este chat (Muestra de inicio, medio y fin).
        Participantes: ${participants.join(', ')}.
        
        Genera un JSON con:
        1. 'personalities': Array de objetos {author, description (max 20 palabras), sentiment (emoji + adjetivo)}.
        2. 'qa': Array de 3 insights clave {question, answer} sobre: Dinámica de poder, Tono emocional, Tema recurrente.
        
        Chat Log:
        ${sampleText.substring(0, 30000)} 
      `;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
        config: {
            systemInstruction: "Eres un psicólogo social experto analizando dinámicas de grupo en WhatsApp. Tu salida debe ser JSON válido estrictamente.",
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
                                description: { type: Type.STRING }, 
                                sentiment: { type: Type.STRING } 
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
                        } 
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
      setError("Error conectando con Gemini. Verifica tu cuota o API Key.");
    } finally {
      setLoading(false);
    }
  };

  if (!result && !loading) {
    return (
      <div style={styles.card}>
         <div style={{textAlign: 'center'}}>
             <h3 style={{color: THEME.primary}}>🧠 Análisis con IA</h3>
             <p style={{color: THEME.subtext, marginBottom: '1rem'}}>Gemini leerá fragmentos clave de la conversación para detectar personalidades.</p>
             <button onClick={runAnalysis} style={{ background: THEME.primary, color: 'white', border: 'none', padding: '0.8rem 2rem', borderRadius: '2rem', cursor: 'pointer', fontSize: '1rem' }}>
                ✨ Analizar ahora
             </button>
         </div>
         {error && <p style={{color: 'red', textAlign: 'center', marginTop: '1rem'}}>{error}</p>}
      </div>
    );
  }

  if (loading) return <div style={{...styles.card, textAlign: 'center'}}>🧠 Pensando... (Esto toma unos segundos)</div>;

  return (
    <div style={styles.grid}>
        <div style={{...styles.card, gridColumn: '1 / -1'}}>
            <h3 style={{marginTop: 0, color: THEME.primary}}>📊 Conclusiones de la IA</h3>
            <div style={{display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))'}}>
                {result?.qa.map((qa, i) => (
                    <div key={i} style={{background: '#f8f9fa', padding: '1rem', borderRadius: '0.5rem', borderLeft: `4px solid ${THEME.primary}`}}>
                        <strong>{qa.question}</strong>
                        <p style={{margin: '0.5rem 0 0', color: '#555'}}>{qa.answer}</p>
                    </div>
                ))}
            </div>
        </div>
        {result?.personalities.map((p, i) => (
            <div key={i} style={{...styles.card, borderTop: `4px solid ${THEME.userColors[i % THEME.userColors.length]}`}}>
                <div style={{display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem'}}>
                    <strong>{p.author}</strong>
                    <span style={{background: '#e8f5e9', padding: '2px 8px', borderRadius: '10px', fontSize: '0.8rem'}}>{p.sentiment}</span>
                </div>
                <p style={{fontSize: '0.9rem', color: THEME.subtext}}>{p.description}</p>
            </div>
        ))}
    </div>
  );
};

// --- NEW COMPONENT: Specialized Hourly Chart ---
const HourlyChart = ({ data, color }: { data: Record<number, number>; color: string }) => {
  // 1. Normalización: Creamos un array fijo de 24 horas (0-23)
  // Esto soluciona el problema de las "horas perdidas" al final del gráfico.
  const hours = Array.from({ length: 24 }, (_, i) => i);
  const values = hours.map(h => data[h] || 0);
  const maxVal = Math.max(...values, 1); // Evitamos división por cero

  // Función para determinar el momento del día (UX visual)
  const getTimeIcon = (h: number) => {
    if (h >= 6 && h < 12) return '🌅'; // Mañana
    if (h >= 12 && h < 19) return '☀️'; // Tarde
    if (h >= 19 || h < 6) return '🌙'; // Noche
    return '';
  };

  return (
    <div style={{ width: '100%', padding: '1rem 0' }}>
      {/* Contenedor del Gráfico */}
      <div style={{ 
        display: 'flex', 
        alignItems: 'flex-end', // CRÍTICO: Hace que las barras crezcan hacia arriba
        height: '200px', 
        gap: '4px',
        borderBottom: '2px solid #e0e0e0',
        paddingBottom: '8px'
      }}>
        {hours.map((hour) => {
          const count = data[hour] || 0;
          const heightPct = (count / maxVal) * 100;
          
          return (
            <div 
              key={hour} 
              style={{ 
                flex: 1, 
                display: 'flex', 
                flexDirection: 'column', 
                justifyContent: 'flex-end',
                height: '100%',
                position: 'relative',
                group: 'bar' // para efectos hover si usáramos CSS puro
              }}
              title={`${hour}:00 - ${count} mensajes`} // Tooltip nativo simple
            >
              {/* La Barra */}
              <div style={{
                height: `${heightPct}%`,
                background: color,
                opacity: heightPct === 0 ? 0.1 : 0.8, // Si es 0, mostramos una sombra tenue para mantener la estructura
                borderRadius: '4px 4px 0 0',
                transition: 'height 0.5s ease',
                minHeight: '4px' // Mínimo visual para que se vea que existe la hora
              }}></div>
            </div>
          );
        })}
      </div>

      {/* Eje X (Etiquetas) */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', color: '#667781', fontSize: '0.75rem', fontWeight: 'bold' }}>
        <span>00:00 🌙</span>
        <span>06:00 🌅</span>
        <span>12:00 ☀️</span>
        <span>18:00 🌇</span>
        <span>23:59 🌙</span>
      </div>
    </div>
  );
};


// --- 5. MAIN APP ---

const App = () => {
  const [messages, setMessages] = useState<Message[] | null>(null);
  
  // Optimizacion: Memoizamos el cálculo de estadísticas.
  // Solo se recalcula si 'messages' cambia.
  const stats = useMemo(() => {
    if (!messages) return null;
    return calculateStats(messages);
  }, [messages]);

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={{ fontSize: '2.5rem', color: THEME.primary, marginBottom: '0.5rem', letterSpacing: '-1px' }}>
          WhatsApp Insights 💬
        </h1>
        <p style={{ color: THEME.subtext, fontSize: '1.1rem' }}>
          Descubre quién habla más, cuándo y qué dicen realmente.
        </p>
      </header>

      {!messages ? (
        <FileUpload onDataParsed={setMessages} />
      ) : stats && (
        <div style={{ display: 'grid', gap: '2rem' }}>
          
          {/* Dashboard Header */}
          <div style={{...styles.card, display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
             <button 
                onClick={() => setMessages(null)}
                style={{ background: 'none', border: 'none', color: THEME.primary, cursor: 'pointer', fontWeight: 'bold' }}
             >
                ← Subir otro
             </button>
             <div style={{ color: THEME.subtext }}>
                 📅 {stats.dateRange.start.toLocaleDateString()} - {stats.dateRange.end.toLocaleDateString()} 
                 <span style={{ margin: '0 10px' }}>|</span>
                 <strong>{stats.totalMessages.toLocaleString()}</strong> mensajes
             </div>
          </div>

          {/* Charts Row 1 */}
          <div style={styles.grid}>
             <div style={styles.card}>
                <h3 style={{ color: THEME.primary, marginTop: 0 }}>🏆 Top Speakers</h3>
                <HorizontalBarList 
                    items={Object.entries(stats.authorCounts).sort(([,a], [,b]) => b - a).map(([k,v]) => ({ label: k, value: v }))} 
                    colorPalette={THEME.userColors} 
                />
             </div>
             <div style={styles.card}>
                <h3 style={{ color: THEME.primary, marginTop: 0 }}>😂 Emojis</h3>
                <HorizontalBarList 
                    items={Object.entries(stats.emojiCounts).sort(([,a], [,b]) => b - a).slice(0, 8).map(([k,v]) => ({ label: k, value: v }))}
                    colorPalette={[THEME.secondary]} 
                />
             </div>
          </div>

          <div style={styles.card}>
            <h3 style={{ color: THEME.primary, marginTop: 0 }}>⏰ Actividad por Hora</h3>
            <p style={{ fontSize: '0.85rem', color: THEME.subtext, marginBottom: '1.5rem' }}>
              Ritmo de conversación durante el día (00h - 23h)
            </p>
    
            {stats && (
              <HourlyChart 
                data={stats.hourlyActivity} 
                color={THEME.primary} 
              />
             )}
          </div>

          {/* AI Section */}          
        </div>
      )}
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);