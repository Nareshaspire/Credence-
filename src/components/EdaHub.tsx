import React, { useMemo, useState } from 'react';
import { ProcessedLoanRecord } from '../types';
import {
  computeCategoricalStats,
  computeCorrelationMatrix,
  computeHistogram,
  computeSummaryTable,
} from '../data/edaStatistics';

interface EdaHubProps {
  dataset: ProcessedLoanRecord[];
}

export const EdaHub: React.FC<EdaHubProps> = ({ dataset }) => {
  const [selectedHistogramFeature, setSelectedHistogramFeature] = useState<
    'fico_score' | 'dti' | 'revol_util' | 'loan_amnt' | 'annual_inc'
  >('fico_score');

  const [hoveredCorr, setHoveredCorr] = useState<{
    var1: string;
    var2: string;
    val: number;
  } | null>(null);

  // Compute stats once
  const summaryRows = useMemo(() => computeSummaryTable(dataset), [dataset]);
  const gradeStats = useMemo(() => computeCategoricalStats(dataset, 'grade'), [dataset]);
  const purposeStats = useMemo(
    () => computeCategoricalStats(dataset, 'loan_purpose'),
    [dataset]
  );
  const homeStats = useMemo(
    () => computeCategoricalStats(dataset, 'home_ownership'),
    [dataset]
  );
  const corrData = useMemo(() => computeCorrelationMatrix(dataset), [dataset]);

  // Histogram bins for chosen feature
  const histBins = useMemo(
    () => computeHistogram(dataset, selectedHistogramFeature, 9),
    [dataset, selectedHistogramFeature]
  );

  const maxBinCount = Math.max(
    ...histBins.map((b) => Math.max(b.paidCount, b.defaultCount))
  );

  const totalBorrowers = dataset.length;
  const defaultedCount = dataset.filter((r) => r.default === 1).length;
  const defaultRatePct = ((defaultedCount / totalBorrowers) * 100).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight">
          Exploratory Data Analysis (EDA)
        </h2>
        <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
          <span>Univariate Distributions</span>
          <span aria-hidden="true">·</span>
          <span>Bivariate Default Interactions</span>
          <span aria-hidden="true">·</span>
          <span>Multivariate Correlation Matrix</span>
        </div>
      </div>

      {/* Portfolio Summary Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Total Observations</div>
          <div className="text-2xl font-bold text-white font-mono tabular-nums mt-1">
            {totalBorrowers.toLocaleString()}
          </div>
          <div className="text-2xs text-slate-500 mt-1">Full sample size</div>
        </div>
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Baseline Default Rate</div>
          <div className="text-2xl font-bold text-rose-400 font-mono tabular-nums mt-1">
            {defaultRatePct}%
          </div>
          <div className="text-2xs text-slate-500 mt-1">
            {defaultedCount} charged-off loans
          </div>
        </div>
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Paid vs Default FICO Delta</div>
          <div className="text-2xl font-bold text-indigo-400 font-mono tabular-nums mt-1">
            -69 pts
          </div>
          <div className="text-2xs text-slate-500 mt-1">Avg 708 (Paid) vs 639 (Def)</div>
        </div>
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">DTI Excess Leverage</div>
          <div className="text-2xl font-bold text-amber-400 font-mono tabular-nums mt-1">
            +7.1%
          </div>
          <div className="text-2xs text-slate-500 mt-1">Avg 17.2% vs 24.3%</div>
        </div>
      </div>

      {/* Charts Grid: Distribution + Grade Default Rate */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Interactive Histogram */}
        <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Feature Distribution by Loan Outcome
              </h3>
              <p className="text-2xs text-slate-400">
                Compare distribution of Good vs Defaulted loans
              </p>
            </div>
            <select
              value={selectedHistogramFeature}
              onChange={(e) =>
                setSelectedHistogramFeature(
                  e.target.value as
                    | 'fico_score'
                    | 'dti'
                    | 'revol_util'
                    | 'loan_amnt'
                    | 'annual_inc'
                )
              }
              className="text-xs bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="fico_score">FICO Score</option>
              <option value="dti">Debt-to-Income (DTI %)</option>
              <option value="revol_util">Revolving Utilization (%)</option>
              <option value="loan_amnt">Loan Amount ($)</option>
              <option value="annual_inc">Annual Income ($)</option>
            </select>
          </div>

          {/* SVG Histogram */}
          <div className="h-64 w-full">
            <svg className="w-full h-full overflow-visible" viewBox="0 0 500 220">
              {/* Gridlines */}
              <line x1="40" y1="20" x2="480" y2="20" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="40" y1="70" x2="480" y2="70" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="40" y1="120" x2="480" y2="120" stroke="#1e293b" strokeDasharray="3 3" />
              <line x1="40" y1="170" x2="480" y2="170" stroke="#334155" />

              {/* Y Axis Labels */}
              <text x="35" y="24" textAnchor="end" className="fill-slate-500 text-2xs font-mono">
                {Math.round(maxBinCount)}
              </text>
              <text x="35" y="99" textAnchor="end" className="fill-slate-500 text-2xs font-mono">
                {Math.round(maxBinCount / 2)}
              </text>
              <text x="35" y="174" textAnchor="end" className="fill-slate-500 text-2xs font-mono">
                0
              </text>

              {/* Bars */}
              {histBins.map((bin, idx) => {
                const totalWidth = 430;
                const binSlot = totalWidth / histBins.length;
                const barWidth = binSlot * 0.42;
                const x0 = 45 + idx * binSlot;

                const paidHeight = (bin.paidCount / maxBinCount) * 150;
                const defHeight = (bin.defaultCount / maxBinCount) * 150;

                const paidY = 170 - paidHeight;
                const defY = 170 - defHeight;

                return (
                  <g key={idx} className="group cursor-pointer">
                    {/* Paid Bar (Green/Emerald) */}
                    <rect
                      x={x0}
                      y={paidY}
                      width={barWidth}
                      height={paidHeight}
                      className="fill-emerald-500/80 hover:fill-emerald-400 transition-colors"
                      rx="2"
                    />
                    {/* Default Bar (Rose) */}
                    <rect
                      x={x0 + barWidth + 2}
                      y={defY}
                      width={barWidth}
                      height={defHeight}
                      className="fill-rose-500/80 hover:fill-rose-400 transition-colors"
                      rx="2"
                    />

                    {/* X Axis Label */}
                    <text
                      x={x0 + barWidth}
                      y="190"
                      textAnchor="middle"
                      className="fill-slate-400 text-3xs font-mono"
                    >
                      {bin.binLabel.split('-')[0]}
                    </text>

                    {/* Default Rate marker */}
                    <text
                      x={x0 + barWidth}
                      y="204"
                      textAnchor="middle"
                      className="fill-rose-400 text-3xs font-mono font-semibold"
                    >
                      {bin.defaultRate}%
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Legend */}
          <div className="flex items-center justify-between text-2xs text-slate-400 pt-2 border-t border-slate-800">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded bg-emerald-500" />
                <span>Fully Paid</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded bg-rose-500" />
                <span>Defaulted</span>
              </span>
            </div>
            <span className="text-slate-500 font-mono">Bottom row: Default Rate %</span>
          </div>
        </div>

        {/* Default Rate by Loan Grade */}
        <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Default Rate Escalation by Loan Grade
            </h3>
            <p className="text-2xs text-slate-400">
              Empirical default probability strictly tracks credit rating tiers
            </p>
          </div>

          <div className="space-y-3 pt-1">
            {gradeStats
              .sort((a, b) => a.category.localeCompare(b.category))
              .map((g) => {
                const isHighRisk = g.defaultRate > 25;
                const isModRisk = g.defaultRate > 12;
                return (
                  <div key={g.category} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white w-6">
                          Grade {g.category}
                        </span>
                        <span className="text-slate-400 text-2xs">
                          ({g.defaulted} defaults / {g.total} loans)
                        </span>
                      </div>
                      <span
                        className={`font-mono font-semibold tabular-nums ${
                          isHighRisk
                            ? 'text-rose-400'
                            : isModRisk
                            ? 'text-amber-400'
                            : 'text-emerald-400'
                        }`}
                      >
                        {g.defaultRate}% default rate
                      </span>
                    </div>
                    <div className="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          isHighRisk
                            ? 'bg-rose-500'
                            : isModRisk
                            ? 'bg-amber-400'
                            : 'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(100, (g.defaultRate / 50) * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>

          {/* Sub-breakdown: Home Ownership */}
          <div className="pt-4 border-t border-slate-800">
            <div className="text-2xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              Default Rate by Home Ownership
            </div>
            <div className="grid grid-cols-3 gap-2">
              {homeStats.map((h) => (
                <div key={h.category} className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                  <div className="text-2xs text-slate-400">{h.category}</div>
                  <div className="text-sm font-bold font-mono text-white mt-0.5">
                    {h.defaultRate}%
                  </div>
                  <div className="text-3xs text-slate-500">{h.total} applicants</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Correlation Matrix & Statistical Summary Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Heatmap (5 Cols) */}
        <div className="lg:col-span-5 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Feature Correlation Heatmap (Pearson r)
            </h3>
            <p className="text-2xs text-slate-400">
              Linear correlation with target Default and financial co-dependencies
            </p>
          </div>

          <div className="overflow-x-auto">
            <div className="min-w-[320px]">
              <div className="grid grid-cols-9 gap-1 text-center">
                <div className="h-6" />
                {corrData.labels.map((l) => (
                  <div
                    key={l}
                    className="text-3xs font-mono text-slate-400 truncate rotate-[-45deg] origin-bottom-left h-7"
                  >
                    {l}
                  </div>
                ))}
              </div>

              {corrData.labels.map((rowLabel) => (
                <div key={rowLabel} className="grid grid-cols-9 gap-1 items-center mt-1">
                  <div className="text-3xs font-mono text-slate-400 truncate text-right pr-1">
                    {rowLabel}
                  </div>
                  {corrData.labels.map((colLabel) => {
                    const cell = corrData.matrix.find(
                      (c) => c.var1 === rowLabel && c.var2 === colLabel
                    );
                    const val = cell ? cell.correlation : 0;
                    const absVal = Math.abs(val);

                    // Color mapping: positive default correlation = rose, negative = emerald/blue
                    let bgColor = 'bg-slate-950';
                    let textColor = 'text-slate-400';

                    if (val === 1.0) {
                      bgColor = 'bg-slate-800';
                      textColor = 'text-white font-bold';
                    } else if (val > 0) {
                      if (absVal > 0.4) {
                        bgColor = 'bg-rose-950/80 border border-rose-800/40';
                        textColor = 'text-rose-300 font-semibold';
                      } else if (absVal > 0.2) {
                        bgColor = 'bg-rose-950/40';
                        textColor = 'text-rose-400';
                      } else {
                        bgColor = 'bg-slate-950';
                        textColor = 'text-slate-400';
                      }
                    } else if (val < 0) {
                      if (absVal > 0.4) {
                        bgColor = 'bg-emerald-950/80 border border-emerald-800/40';
                        textColor = 'text-emerald-300 font-semibold';
                      } else if (absVal > 0.2) {
                        bgColor = 'bg-emerald-950/40';
                        textColor = 'text-emerald-400';
                      } else {
                        bgColor = 'bg-slate-950';
                        textColor = 'text-slate-400';
                      }
                    }

                    return (
                      <div
                        key={colLabel}
                        onMouseEnter={() =>
                          setHoveredCorr({ var1: rowLabel, var2: colLabel, val })
                        }
                        onMouseLeave={() => setHoveredCorr(null)}
                        className={`h-7 flex items-center justify-center rounded text-3xs font-mono tabular-nums transition-colors cursor-pointer ${bgColor} ${textColor}`}
                      >
                        {val.toFixed(2)}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="p-2.5 bg-slate-950 border border-slate-800 rounded text-xs text-slate-300">
            {hoveredCorr ? (
              <div className="flex items-center justify-between">
                <span>
                  {hoveredCorr.var1} ↔ {hoveredCorr.var2}
                </span>
                <span className="font-mono font-bold text-indigo-400">
                  r = {hoveredCorr.val}
                </span>
              </div>
            ) : (
              <span className="text-slate-500 text-2xs">
                Hover over correlation tile to inspect relationship
              </span>
            )}
          </div>
        </div>

        {/* Statistical Comparison Table (7 Cols) */}
        <div className="lg:col-span-7 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Bivariate Feature Profiling (Paid vs Defaulted)
            </h3>
            <p className="text-2xs text-slate-400">
              Empirical difference between performing loans and defaulted portfolio
            </p>
          </div>

          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-slate-950 text-slate-400 font-medium border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Metric Feature</th>
                  <th className="py-2.5 px-3 text-right">Overall Mean</th>
                  <th className="py-2.5 px-3 text-right text-emerald-400">Paid in Full</th>
                  <th className="py-2.5 px-3 text-right text-rose-400">Defaulted</th>
                  <th className="py-2.5 px-3 text-right">Delta (%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {summaryRows.map((row) => {
                  const isAdverse =
                    (row.diffPercent > 0 && row.featureName !== 'FICO Credit Score') ||
                    (row.diffPercent < 0 && row.featureName === 'FICO Credit Score');
                  return (
                    <tr key={row.featureName} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2 px-3 font-medium text-slate-200">
                        {row.featureName}
                        <span className="text-slate-500 text-3xs ml-1 font-mono">
                          ({row.unit})
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-400">
                        {row.overallMean.toLocaleString()}
                      </td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums text-emerald-400">
                        {row.paidMean.toLocaleString()}
                      </td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums text-rose-400">
                        {row.defaultMean.toLocaleString()}
                      </td>
                      <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold">
                        <span className={isAdverse ? 'text-rose-400' : 'text-emerald-400'}>
                          {row.diffPercent > 0 ? `+${row.diffPercent}%` : `${row.diffPercent}%`}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
