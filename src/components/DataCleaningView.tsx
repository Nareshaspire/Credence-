import React, { useState } from 'react';
import { ProcessedLoanRecord } from '../types';
import { DatasetSplit } from '../ml/featureEngineering';

interface DataCleaningViewProps {
  dataset: ProcessedLoanRecord[];
  split: DatasetSplit;
}

export const DataCleaningView: React.FC<DataCleaningViewProps> = ({ dataset, split }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDefault, setFilterDefault] = useState<'all' | '0' | '1'>('all');
  const [page, setPage] = useState(0);
  const pageSize = 12;

  const filtered = dataset.filter((r) => {
    const matchesSearch =
      r.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.loan_purpose.toLowerCase().includes(searchTerm.toLowerCase()) ||
      r.grade.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDefault =
      filterDefault === 'all'
        ? true
        : filterDefault === '1'
        ? r.default === 1
        : r.default === 0;
    return matchesSearch && matchesDefault;
  });

  const paginated = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(filtered.length / pageSize);

  const downloadCSV = () => {
    const headers = [
      'id',
      'loan_amnt',
      'term_months',
      'int_rate',
      'grade',
      'annual_inc',
      'emp_length_years',
      'home_ownership',
      'fico_score',
      'dti',
      'revol_util',
      'total_acc',
      'delinq_2yrs',
      'inq_last_6mths',
      'loan_purpose',
      'installment_burden',
      'credit_stability_index',
      'default',
    ];
    const rows = dataset.map((r) => [
      r.id,
      r.loan_amnt,
      r.term_months,
      r.int_rate,
      r.grade,
      r.annual_inc,
      r.emp_length_years,
      r.home_ownership,
      r.fico_score,
      r.dti,
      r.revol_util,
      r.total_acc,
      r.delinq_2yrs,
      r.inq_last_6mths,
      r.loan_purpose,
      r.installment_burden,
      r.credit_stability_index,
      r.default,
    ]);
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'cleaned_lending_club_dataset.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Overview Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Data Cleaning & Preprocessing Pipeline
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
            <span>Source: Empirical LendingClub Credit Schema</span>
            <span aria-hidden="true">·</span>
            <span>1,200 Observations</span>
            <span aria-hidden="true">·</span>
            <span>18 Raw & Engineered Features</span>
          </div>
        </div>
        <button
          onClick={downloadCSV}
          className="px-4 py-2 text-xs font-semibold text-slate-200 bg-slate-800 border border-slate-700 rounded-lg hover:bg-slate-700 transition-colors whitespace-nowrap self-start sm:self-auto"
        >
          Download Cleaned CSV
        </button>
      </div>

      {/* Preprocessing Architecture Steps */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2">
          <div className="text-xs font-semibold text-indigo-400">01. Ingestion & Quality</div>
          <div className="text-sm font-medium text-slate-200">Zero Missing Values</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            All records validated with median imputation on employment length and mode imputation on home ownership.
          </p>
          <div className="text-xs font-mono text-emerald-400 pt-1">100% Completeness</div>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2">
          <div className="text-xs font-semibold text-indigo-400">02. Outlier Treatment</div>
          <div className="text-sm font-medium text-slate-200">Winsorization & Bounds</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Annual income capped at 99th percentile ($260,000) and DTI capped at 45.0% to prevent leverage distortion.
          </p>
          <div className="text-xs font-mono text-emerald-400 pt-1">Robust Scaling Applied</div>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2">
          <div className="text-xs font-semibold text-indigo-400">03. Feature Engineering</div>
          <div className="text-sm font-medium text-slate-200">Financial Risk Indices</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Synthesized Payment Burden %, Credit Stability Index, and DTI × FICO Deficit interaction multiplier.
          </p>
          <div className="text-xs font-mono text-emerald-400 pt-1">+4 Domain Features</div>
        </div>

        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2">
          <div className="text-xs font-semibold text-indigo-400">04. Stratified Split</div>
          <div className="text-sm font-medium text-slate-200">80 / 20 Train-Test Ratio</div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Stratified by loan default target. Preserves true empirical 18.5% default base rate across both sets.
          </p>
          <div className="text-xs font-mono text-emerald-400 pt-1">960 Train / 240 Test</div>
        </div>
      </div>

      {/* Dataset Filter & Search Bar */}
      <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <input
              type="text"
              placeholder="Search by ID, purpose, or grade..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setPage(0);
              }}
              className="px-3.5 py-2 text-xs bg-slate-950 border border-slate-800 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 w-64"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="text-xs text-slate-400 hover:text-white"
              >
                Clear
              </button>
            )}
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-950 border border-slate-800 rounded-lg">
            <button
              onClick={() => {
                setFilterDefault('all');
                setPage(0);
              }}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                filterDefault === 'all'
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              All Records ({dataset.length})
            </button>
            <button
              onClick={() => {
                setFilterDefault('0');
                setPage(0);
              }}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                filterDefault === '0'
                  ? 'bg-slate-800 text-emerald-400'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Fully Paid ({dataset.filter((r) => r.default === 0).length})
            </button>
            <button
              onClick={() => {
                setFilterDefault('1');
                setPage(0);
              }}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                filterDefault === '1'
                  ? 'bg-slate-800 text-rose-400'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Defaulted ({dataset.filter((r) => r.default === 1).length})
            </button>
          </div>
        </div>

        {/* Tabular Grid */}
        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-950 text-slate-400 font-medium border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Loan ID</th>
                <th className="py-2.5 px-3 text-right">Amount</th>
                <th className="py-2.5 px-3 text-right">Rate</th>
                <th className="py-2.5 px-3">Grade</th>
                <th className="py-2.5 px-3 text-right">Income</th>
                <th className="py-2.5 px-3 text-right">FICO</th>
                <th className="py-2.5 px-3 text-right">DTI</th>
                <th className="py-2.5 px-3 text-right">Revol Util</th>
                <th className="py-2.5 px-3 text-right">Delinq</th>
                <th className="py-2.5 px-3 text-right">Inquiries</th>
                <th className="py-2.5 px-3">Purpose</th>
                <th className="py-2.5 px-3 text-center">Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {paginated.length > 0 ? (
                paginated.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-800/50 transition-colors">
                    <td className="py-2 px-3 font-mono text-slate-400">{r.id}</td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      ${r.loan_amnt.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {r.int_rate}%
                    </td>
                    <td className="py-2 px-3">
                      <span className="font-semibold text-slate-200">{r.grade}</span>
                      <span className="text-slate-500 text-2xs ml-1">{r.sub_grade}</span>
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      ${r.annual_inc.toLocaleString()}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums font-semibold text-slate-200">
                      {r.fico_score}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {r.dti}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {r.revol_util}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {r.delinq_2yrs}
                    </td>
                    <td className="py-2 px-3 text-right font-mono tabular-nums">
                      {r.inq_last_6mths}
                    </td>
                    <td className="py-2 px-3 text-slate-400 capitalize">
                      {r.loan_purpose.replace('_', ' ')}
                    </td>
                    <td className="py-2 px-3 text-center">
                      <span
                        className={`font-semibold text-2xs uppercase ${
                          r.default === 1 ? 'text-rose-400' : 'text-emerald-400'
                        }`}
                      >
                        {r.default === 1 ? 'Defaulted' : 'Paid in Full'}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={12} className="py-8 text-center text-slate-500">
                    No loan records match the selected criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination controls */}
        <div className="flex items-center justify-between text-xs text-slate-400 pt-2">
          <div>
            Showing <span className="font-mono text-slate-200">{filtered.length === 0 ? 0 : page * pageSize + 1}</span> to{' '}
            <span className="font-mono text-slate-200">
              {Math.min((page + 1) * pageSize, filtered.length)}
            </span>{' '}
            of <span className="font-mono text-slate-200">{filtered.length}</span> rows
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-3 py-1 bg-slate-950 border border-slate-800 rounded disabled:opacity-40 hover:bg-slate-800 transition-colors"
            >
              Previous
            </button>
            <span className="font-mono text-slate-300">
              Page {page + 1} of {Math.max(1, totalPages)}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-3 py-1 bg-slate-950 border border-slate-800 rounded disabled:opacity-40 hover:bg-slate-800 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
