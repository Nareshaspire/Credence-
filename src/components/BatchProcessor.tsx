import React, { useState, useMemo } from 'react';
import { ModelMetrics, ProcessedLoanRecord } from '../types';
import { MLModel } from '../ml/algorithms';
import { DatasetSplit, extractFeatureVector, transformFeatures } from '../ml/featureEngineering';

interface BatchProcessorProps {
  dataset: ProcessedLoanRecord[];
  split: DatasetSplit;
  activeModel: MLModel;
  activeMetrics: ModelMetrics;
}

export const BatchProcessor: React.FC<BatchProcessorProps> = ({
  dataset,
  split,
  activeModel,
  activeMetrics,
}) => {
  const [batchSize, setBatchSize] = useState<number>(30);
  const [searchTerm, setSearchTerm] = useState<string>('');

  // Sample batch from test partition
  const batchRecords = useMemo(() => {
    return split.testRecords.slice(0, batchSize);
  }, [split.testRecords, batchSize]);

  // Score batch using active model
  const scoredBatch = useMemo(() => {
    const rawX = batchRecords.map(extractFeatureVector);
    const normX = transformFeatures(rawX, split.scalingMap);
    const probas = activeModel.predictProba(normX);
    const threshold = activeMetrics.hyperparameters.decisionThreshold;

    return batchRecords.map((r, idx) => {
      const prob = probas[idx];
      const isApproved = prob < threshold;
      let tier = 'BBB';
      if (prob < 0.05) tier = 'AAA';
      else if (prob < 0.1) tier = 'AA';
      else if (prob < 0.16) tier = 'A';
      else if (prob < 0.25) tier = 'BBB';
      else if (prob < 0.36) tier = 'BB';
      else if (prob < 0.5) tier = 'B';
      else tier = 'D';

      const expLoss = Math.round(r.loan_amnt * prob * 0.7);

      return {
        ...r,
        predictedProb: parseFloat((prob * 100).toFixed(1)),
        isApproved,
        riskTier: tier,
        expectedLoss: expLoss,
      };
    });
  }, [batchRecords, activeModel, activeMetrics, split.scalingMap]);

  // Portfolio aggregates
  const totalVolume = scoredBatch.reduce((acc, r) => acc + r.loan_amnt, 0);
  const approvedVolume = scoredBatch
    .filter((r) => r.isApproved)
    .reduce((acc, r) => acc + r.loan_amnt, 0);
  const approvedCount = scoredBatch.filter((r) => r.isApproved).length;
  const avgDefaultRisk =
    scoredBatch.reduce((acc, r) => acc + r.predictedProb, 0) / (scoredBatch.length || 1);
  const totalExpLoss = scoredBatch.reduce((acc, r) => acc + r.expectedLoss, 0);

  const filteredBatch = scoredBatch.filter(
    (r) =>
      r.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.loan_purpose.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const exportScoredCSV = () => {
    const headers = [
      'id',
      'loan_amnt',
      'annual_inc',
      'fico_score',
      'dti',
      'revol_util',
      'purpose',
      'predicted_default_prob_pct',
      'approval_status',
      'risk_tier',
      'expected_loss_usd',
    ];

    const rows = scoredBatch.map((r) => [
      r.id,
      r.loan_amnt,
      r.annual_inc,
      r.fico_score,
      r.dti,
      r.revol_util,
      r.loan_purpose,
      r.predictedProb,
      r.isApproved ? 'APPROVED' : 'DECLINED',
      r.riskTier,
      r.expectedLoss,
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `scored_loan_portfolio_${batchSize}_records.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Batch Credit Inference & Portfolio Assessment
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
            <span>Model: {activeMetrics.algorithmName}</span>
            <span aria-hidden="true">·</span>
            <span>Batch Pipeline Size: {batchSize} Applications</span>
            <span aria-hidden="true">·</span>
            <span>Decision Threshold: {(activeMetrics.hyperparameters.decisionThreshold * 100).toFixed(0)}%</span>
          </div>
        </div>
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <select
            value={batchSize}
            onChange={(e) => setBatchSize(parseInt(e.target.value))}
            className="text-xs bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-indigo-500"
          >
            <option value={15}>Batch: 15 Applications</option>
            <option value={30}>Batch: 30 Applications</option>
            <option value={60}>Batch: 60 Applications</option>
            <option value={120}>Batch: 120 Applications</option>
          </select>
          <button
            onClick={exportScoredCSV}
            className="px-4 py-2 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors whitespace-nowrap"
          >
            Export Scored CSV
          </button>
        </div>
      </div>

      {/* Aggregated Portfolio Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Total Capital Requested</div>
          <div className="text-2xl font-bold font-mono text-white mt-1 tabular-nums">
            ${totalVolume.toLocaleString()}
          </div>
          <div className="text-3xs text-slate-500 mt-1">{scoredBatch.length} applicants in batch</div>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Cohort Approval Rate</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1 tabular-nums">
            {((approvedCount / (scoredBatch.length || 1)) * 100).toFixed(1)}%
          </div>
          <div className="text-3xs text-slate-500 mt-1">
            {approvedCount} approved (${approvedVolume.toLocaleString()})
          </div>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Mean Default Probability</div>
          <div className="text-2xl font-bold font-mono text-indigo-400 mt-1 tabular-nums">
            {avgDefaultRisk.toFixed(1)}%
          </div>
          <div className="text-3xs text-slate-500 mt-1">Underwritten portfolio average</div>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl">
          <div className="text-xs text-slate-400">Projected Credit Loss</div>
          <div className="text-2xl font-bold font-mono text-amber-400 mt-1 tabular-nums">
            ${totalExpLoss.toLocaleString()}
          </div>
          <div className="text-3xs text-slate-500 mt-1">At 70% Loss-Given-Default</div>
        </div>
      </div>

      {/* Batch Table */}
      <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <h3 className="text-sm font-semibold text-white">Scored Loan Applicants</h3>
          <input
            type="text"
            placeholder="Filter by ID or purpose..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="px-3 py-1.5 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-56"
          />
        </div>

        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-950 text-slate-400 font-medium border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Loan ID</th>
                <th className="py-2.5 px-3 text-right">Amount</th>
                <th className="py-2.5 px-3 text-right">Income</th>
                <th className="py-2.5 px-3 text-right">FICO</th>
                <th className="py-2.5 px-3 text-right">DTI</th>
                <th className="py-2.5 px-3">Purpose</th>
                <th className="py-2.5 px-3 text-right">Default Prob</th>
                <th className="py-2.5 px-3 text-center">Tier</th>
                <th className="py-2.5 px-3 text-right">Expected Loss</th>
                <th className="py-2.5 px-3 text-center">Disposition</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {filteredBatch.map((r) => (
                <tr key={r.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-2 px-3 font-mono text-slate-400">{r.id}</td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    ${r.loan_amnt.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-400">
                    ${r.annual_inc.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold text-slate-200">
                    {r.fico_score}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums">
                    {r.dti}%
                  </td>
                  <td className="py-2 px-3 text-slate-400 capitalize">
                    {r.loan_purpose.replace('_', ' ')}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums font-bold">
                    <span
                      className={
                        r.predictedProb > 30
                          ? 'text-rose-400'
                          : r.predictedProb > 15
                          ? 'text-amber-400'
                          : 'text-emerald-400'
                      }
                    >
                      {r.predictedProb}%
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center font-mono font-bold text-white">
                    {r.riskTier}
                  </td>
                  <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-400">
                    ${r.expectedLoss.toLocaleString()}
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span
                      className={`font-semibold text-2xs uppercase ${
                        r.isApproved ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {r.isApproved ? 'Approved' : 'Declined'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
