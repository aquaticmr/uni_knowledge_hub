import ChatInterface from "./components/ChatInterface";

function App() {
  return (
    <div className="min-h-screen bg-radial-grid px-4 py-6 md:px-8 md:py-8">
      <div className="mx-auto flex w-full max-w-7xl flex-col overflow-hidden rounded-3xl border border-rbu-100/70 bg-white/90 shadow-panel backdrop-blur md:min-h-[86vh] md:flex-row">
        <aside className="relative overflow-hidden border-b border-rbu-100 bg-rbu-900 px-6 py-6 text-white md:w-80 md:border-b-0 md:border-r md:px-7 md:py-8">
          <div className="pointer-events-none absolute -right-14 -top-16 h-52 w-52 rounded-full bg-rbu-500/25 blur-2xl" />
          <div className="pointer-events-none absolute -bottom-16 -left-14 h-48 w-48 rounded-full bg-rbu-200/20 blur-2xl" />

          <p className="text-xs uppercase tracking-[0.35em] text-rbu-200">India</p>
          <h1 className="mt-3 font-display text-3xl leading-tight text-white">University Knowledge Hub</h1>
          <p className="mt-3 text-sm leading-6 text-rbu-100/90">
            Ask about admissions, fees, hostel, placement stats, recruiters, and eligibility.
          </p>

          <div className="mt-8 rounded-2xl border border-rbu-400/35 bg-rbu-800/70 p-4">
            <p className="text-xs uppercase tracking-[0.22em] text-rbu-200">Powered by</p>
            <p className="mt-1 text-sm font-semibold text-white">FastAPI + RAG + ChromaDB</p>
          </div>
        </aside>

        <main className="flex min-h-[70vh] flex-1 bg-gradient-to-b from-white to-rbu-50/30">
          <ChatInterface />
        </main>
      </div>
    </div>
  );
}

export default App;
