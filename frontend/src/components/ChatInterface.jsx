import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

function ChatInterface() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello. I am your RBU assistant. Ask me about admissions, fees, placements, cutoff trends, or hostel information."
    }
  ]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({ total_chunks: 0, status: "loading" });
  const chatEndRef = useRef(null);

  const chunkLabel = useMemo(() => {
    if (stats.status === "loading") return "Loading...";
    return String(stats.total_chunks ?? 0);
  }, [stats]);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        setStats(data);
      } catch (error) {
        setStats({ total_chunks: 0, status: "offline" });
      }
    };

    fetchStats();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendQuestion = async (event) => {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) {
      return;
    }

    const userMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed })
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer || "No answer returned." }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "I could not reach the backend right now. Please verify FastAPI is running on http://127.0.0.1:8000."
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="flex w-full flex-col">
      <header className="border-b border-rbu-100/80 bg-white/90 px-5 py-4 md:px-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-xl text-rbu-900 md:text-2xl">Admission and Placement Assistant</h2>
            <p className="text-sm text-rbu-700/80">Real-time answers from the RBU knowledge base</p>
          </div>
          <div className="rounded-xl border border-rbu-200 bg-rbu-50 px-4 py-2 text-right">
            <p className="text-[11px] uppercase tracking-[0.22em] text-rbu-600">Scanned Chunks</p>
            <p className="font-display text-2xl text-rbu-900">{chunkLabel}</p>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-4 py-5 md:px-8 md:py-6">
        <div className="mx-auto flex w-full max-w-4xl flex-col gap-4">
          {messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`max-w-[92%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm md:max-w-[78%] ${
                message.role === "user"
                  ? "ml-auto bg-rbu-800 text-white"
                  : "mr-auto border border-rbu-100 bg-white text-rbu-900"
              }`}
            >
              {message.content}
            </article>
          ))}

          {loading && (
            <div className="mr-auto inline-flex items-center gap-2 rounded-2xl border border-rbu-100 bg-white px-4 py-3 text-sm text-rbu-700">
              <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-rbu-700" />
              <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-rbu-600 [animation-delay:120ms]" />
              <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-rbu-500 [animation-delay:240ms]" />
              Thinking...
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
      </div>

      <form onSubmit={sendQuestion} className="border-t border-rbu-100/80 bg-white/95 px-4 py-4 md:px-8 md:py-5">
        <div className="mx-auto flex w-full max-w-4xl gap-3">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask your question about admissions, fees, or placements..."
            className="w-full rounded-xl border border-rbu-200 bg-white px-4 py-3 text-sm text-rbu-900 outline-none transition focus:border-rbu-500 focus:ring-2 focus:ring-rbu-200"
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="rounded-xl bg-rbu-800 px-5 py-3 text-sm font-semibold text-white transition hover:bg-rbu-700 disabled:cursor-not-allowed disabled:bg-rbu-400"
          >
            Send
          </button>
        </div>
      </form>
    </section>
  );
}

export default ChatInterface;
