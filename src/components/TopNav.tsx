import React from 'react';

export type ActiveTab = 'eda' | 'cleaning' | 'models' | 'predict' | 'batch' | 'code';

interface TopNavProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  onQuickPredict: () => void;
}

export const TopNav: React.FC<TopNavProps> = ({
  activeTab,
  setActiveTab,
  onQuickPredict,
}) => {
  return (
    <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-slate-900 border-b border-slate-800 text-slate-100">
      {/* Zone 1: Brand Wordmark (Single text element) */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white text-base shadow-sm">
          C
        </div>
        <span className="text-base font-bold tracking-tight text-white whitespace-nowrap">
          Credence Risk ML
        </span>
      </div>

      {/* Zone 2: Navigation Links (single-line, concise) */}
      <nav className="hidden lg:flex items-center gap-6 text-sm font-medium text-slate-300">
        <button
          onClick={() => setActiveTab('eda')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'eda'
              ? 'text-white border-b-2 border-indigo-500 font-semibold'
              : 'text-slate-400'
          }`}
        >
          Exploratory Data Analysis
        </button>
        <button
          onClick={() => setActiveTab('cleaning')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'cleaning'
              ? 'text-white border-b-2 border-indigo-500 font-semibold'
              : 'text-slate-400'
          }`}
        >
          Data Preprocessing
        </button>
        <button
          onClick={() => setActiveTab('models')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'models'
              ? 'text-white border-b-2 border-indigo-500 font-semibold'
              : 'text-slate-400'
          }`}
        >
          Model Benchmark & Tuning
        </button>
        <button
          onClick={() => setActiveTab('predict')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'predict'
              ? 'text-white border-b-2 border-indigo-500 font-semibold'
              : 'text-slate-400'
          }`}
        >
          Loan Underwriting Sandbox
        </button>
        <button
          onClick={() => setActiveTab('batch')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'batch'
              ? 'text-white border-b-2 border-indigo-500 font-semibold'
              : 'text-slate-400'
          }`}
        >
          Batch Processing
        </button>
        <button
          onClick={() => setActiveTab('code')}
          className={`hover:text-white transition-colors whitespace-nowrap shrink-0 py-1 ${
            activeTab === 'code'
              ? 'text-white border-b-2 border-indigo-500 font-semibold'
              : 'text-slate-400'
          }`}
        >
          FastAPI & Streamlit Code
        </button>
      </nav>

      {/* Zone 3: 1 Primary Action */}
      <div className="flex items-center gap-3 shrink-0">
        <button
          onClick={onQuickPredict}
          className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors whitespace-nowrap shadow-sm"
        >
          Test Applicant
        </button>
      </div>
    </header>
  );
};
