import React, { useState, useMemo } from 'react';
import { ApplicantFormInput, HomeOwnership, LoanPurpose, ModelAlgorithm, ModelMetrics } from '../types';
import { MLModel } from '../ml/algorithms';
import { DatasetSplit, transformApplicantInput } from '../ml/featureEngineering';
import { generateApplicantPrediction } from '../ml/evaluator';

interface PredictionInterfaceProps {
  split: DatasetSplit;
  activeModel: MLModel;
  activeMetrics: ModelMetrics;
  allModels: Record<ModelAlgorithm, MLModel>;
}

const PRESET_PERSONAS: { name: string; desc: string; input: ApplicantFormInput }[] = [
  {
    name: 'Prime Tech Lead',
    desc: 'High income, exceptional credit, low leverage',
    input: {
      loan_amnt: 25000,
      term_months: 36,
      annual_inc: 165000,
      emp_length_years: 7,
      home_ownership: 'MORTGAGE',
      fico_score: 790,
      dti: 11.5,
      revol_util: 18.0,
      total_acc: 24,
      delinq_2yrs: 0,
      inq_last_6mths: 0,
      loan_purpose: 'debt_consolidation',
    },
  },
  {
    name: 'Near-Prime Family',
    desc: 'Solid income with moderate debt burden',
    input: {
      loan_amnt: 15000,
      term_months: 36,
      annual_inc: 72000,
      emp_length_years: 4,
      home_ownership: 'RENT',
      fico_score: 675,
      dti: 24.5,
      revol_util: 52.0,
      total_acc: 16,
      delinq_2yrs: 0,
      inq_last_6mths: 1,
      loan_purpose: 'home_improvement',
    },
  },
  {
    name: 'Subprime Distressed',
    desc: 'High credit utilization, recent delinquencies',
    input: {
      loan_amnt: 18000,
      term_months: 60,
      annual_inc: 38000,
      emp_length_years: 2,
      home_ownership: 'RENT',
      fico_score: 605,
      dti: 38.0,
      revol_util: 89.0,
      total_acc: 11,
      delinq_2yrs: 2,
      inq_last_6mths: 3,
      loan_purpose: 'debt_consolidation',
    },
  },
  {
    name: 'Commercial Entrepreneur',
    desc: 'Small business purpose with good credit rating',
    input: {
      loan_amnt: 32000,
      term_months: 60,
      annual_inc: 98000,
      emp_length_years: 5,
      home_ownership: 'MORTGAGE',
      fico_score: 725,
      dti: 19.0,
      revol_util: 38.0,
      total_acc: 20,
      delinq_2yrs: 0,
      inq_last_6mths: 1,
      loan_purpose: 'small_business',
    },
  },
];

export const PredictionInterface: React.FC<PredictionInterfaceProps> = ({
  split,
  activeModel,
  activeMetrics,
}) => {
  const [form, setForm] = useState<ApplicantFormInput>(PRESET_PERSONAS[0].input);
  const [whatIfScenario, setWhatIfScenario] = useState<{
    label: string;
    patch: Partial<ApplicantFormInput>;
  } | null>(null);

  const [aiMemo, setAiMemo] = useState<string | null>(null);
  const [isGeneratingMemo, setIsGeneratingMemo] = useState(false);

  // Compute live prediction
  const effectiveForm = useMemo(() => {
    return whatIfScenario ? { ...form, ...whatIfScenario.patch } : form;
  }, [form, whatIfScenario]);

  const { normalizedVector } = useMemo(
    () => transformApplicantInput(effectiveForm, split.scalingMap),
    [effectiveForm, split.scalingMap]
  );

  const prediction = useMemo(
    () =>
      generateApplicantPrediction(
        activeModel,
        normalizedVector,
        effectiveForm,
        activeMetrics.hyperparameters.decisionThreshold
      ),
    [activeModel, normalizedVector, effectiveForm, activeMetrics.hyperparameters.decisionThreshold]
  );

  // Baseline prediction (without what-if)
  const baselinePrediction = useMemo(() => {
    if (!whatIfScenario) return null;
    const { normalizedVector: baseVec } = transformApplicantInput(form, split.scalingMap);
    return generateApplicantPrediction(
      activeModel,
      baseVec,
      form,
      activeMetrics.hyperparameters.decisionThreshold
    );
  }, [whatIfScenario, form, split.scalingMap, activeModel, activeMetrics.hyperparameters.decisionThreshold]);

  const handlePersonaSelect = (persona: typeof PRESET_PERSONAS[0]) => {
    setForm(persona.input);
    setWhatIfScenario(null);
    setAiMemo(null);
  };

  // Generate AI Underwriting Memorandum
  const generateUnderwritingMemo = async () => {
    setIsGeneratingMemo(true);
    try {
      // Create local contextual loan underwriting memo
      const defaultRiskPercent = (prediction.defaultProbability * 100).toFixed(1);
      const isApproved = prediction.recommendation === 'Approved';
      const isDeclined = prediction.recommendation === 'High Risk Declined';

      await new Promise((res) => setTimeout(res, 400));

      const memo = `
CREDENCE CREDIT RISK COMMITTEE — UNDERWRITING MEMORANDUM

APPLICANT PROFILE:
• Requested Principal: $${effectiveForm.loan_amnt.toLocaleString()} (${effectiveForm.term_months} Months)
• Stated Purpose: ${effectiveForm.loan_purpose.replace('_', ' ').toUpperCase()}
• Annual Income: $${effectiveForm.annual_inc.toLocaleString()} | Verified DTI: ${effectiveForm.dti}%
• Bureau Credit Score: ${effectiveForm.fico_score} FICO | Revolving Line Utilization: ${effectiveForm.revol_util}%
• Adverse History: ${effectiveForm.delinq_2yrs} Delinquency events (24m), ${effectiveForm.inq_last_6mths} Hard Inquiries (6m)

ALGORITHMIC RISK ASSESSMENT (${activeMetrics.algorithmName}):
• Predicted Probability of Default: ${defaultRiskPercent}%
• Calibrated Credit Grade: Tier ${prediction.riskTier}
• Underwriting Disposition: ${prediction.recommendation.toUpperCase()}
• Recommended Risk-Adjusted APR: ${prediction.recommendedRate.toFixed(2)}%
• Expected Loss (LGD 70%): $${prediction.expectedLoss.toLocaleString()}

PRIMARY RISK FACTORS & COVENANTS:
${
  isDeclined
    ? `1. High credit risk exceeds portfolio risk tolerance threshold (${(activeMetrics.hyperparameters.decisionThreshold * 100).toFixed(0)}%).
2. Primary adverse drivers: Elevated DTI (${effectiveForm.dti}%) combined with ${effectiveForm.delinq_2yrs} historical delinquency events.
3. Mitigation requirement: Requires credit repair, debt paydown to sub-20% DTI, or adding an eligible co-borrower with FICO ≥ 720.`
    : isApproved
    ? `1. Robust debt service coverage with low payment burden relative to monthly income.
2. Exemplary revolving line management (${effectiveForm.revol_util}%) and zero severe derogatory marks.
3. Underwriting condition: Standard ACH autopay enrollment recommended for prime rate eligibility.`
    : `1. Moderate credit risk requiring senior underwriter sign-off.
2. Direct payoff requirement: Loan funds must be disbursed directly to existing creditors to ensure debt consolidation.
3. Verification condition: 2 months of consecutive pay stubs and proof of employment continuity.`
}
      `.trim();

      setAiMemo(memo);
    } catch (err) {
      console.error(err);
    } finally {
      setIsGeneratingMemo(false);
    }
  };

  const isApproved = prediction.recommendation === 'Approved';
  const isDeclined = prediction.recommendation === 'High Risk Declined';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Interactive Loan Underwriting Sandbox
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
            <span>Inference Model: {activeMetrics.algorithmName}</span>
            <span aria-hidden="true">·</span>
            <span>Decision Cutoff: {(activeMetrics.hyperparameters.decisionThreshold * 100).toFixed(0)}%</span>
            <span aria-hidden="true">·</span>
            <span>Real-time Explainable AI Waterfall</span>
          </div>
        </div>
      </div>

      {/* Preset Personas Strip */}
      <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
        <div className="text-xs font-semibold text-slate-300">
          Load Sample Borrower Archetypes
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {PRESET_PERSONAS.map((p) => {
            const isSelected = form.fico_score === p.input.fico_score && form.annual_inc === p.input.annual_inc;
            return (
              <button
                key={p.name}
                onClick={() => handlePersonaSelect(p)}
                className={`p-3 text-left rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-indigo-950/60 border-indigo-500 shadow-sm'
                    : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="text-xs font-bold text-white">{p.name}</div>
                <div className="text-3xs text-slate-400 mt-0.5 leading-snug">{p.desc}</div>
                <div className="text-3xs font-mono text-slate-500 mt-1.5">
                  FICO {p.input.fico_score} · ${p.input.loan_amnt.toLocaleString()}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Grid: Form on Left (5 Cols), Realtime Scoring on Right (7 Cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Underwriting Form Controls */}
        <div className="lg:col-span-5 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <h3 className="text-sm font-semibold text-white">Applicant Financial Variables</h3>

          <div className="space-y-4 text-xs">
            {/* Loan Amount */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-slate-300">Loan Amount Requested:</label>
                <span className="font-mono text-white font-bold tabular-nums">
                  ${form.loan_amnt.toLocaleString()}
                </span>
              </div>
              <input
                type="range"
                min="1000"
                max="40000"
                step="500"
                value={form.loan_amnt}
                onChange={(e) => setForm({ ...form, loan_amnt: parseInt(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Term & Purpose */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-400 block mb-1">Loan Term:</label>
                <select
                  value={form.term_months}
                  onChange={(e) =>
                    setForm({ ...form, term_months: parseInt(e.target.value) as 36 | 60 })
                  }
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value={36}>36 Months (3 Yrs)</option>
                  <option value={60}>60 Months (5 Yrs)</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Loan Purpose:</label>
                <select
                  value={form.loan_purpose}
                  onChange={(e) =>
                    setForm({ ...form, loan_purpose: e.target.value as LoanPurpose })
                  }
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  <option value="debt_consolidation">Debt Consolidation</option>
                  <option value="credit_card">Credit Card Refinance</option>
                  <option value="home_improvement">Home Improvement</option>
                  <option value="small_business">Small Business</option>
                  <option value="major_purchase">Major Purchase</option>
                  <option value="medical">Medical Expenses</option>
                </select>
              </div>
            </div>

            {/* FICO Score */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-slate-300">FICO Credit Score (580 - 850):</label>
                <span className="font-mono text-indigo-400 font-bold tabular-nums">
                  {form.fico_score}
                </span>
              </div>
              <input
                type="range"
                min="580"
                max="850"
                step="5"
                value={form.fico_score}
                onChange={(e) => setForm({ ...form, fico_score: parseInt(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Annual Income */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-slate-300">Annual Gross Income ($):</label>
                <span className="font-mono text-white font-bold tabular-nums">
                  ${form.annual_inc.toLocaleString()}
                </span>
              </div>
              <input
                type="range"
                min="20000"
                max="250000"
                step="2500"
                value={form.annual_inc}
                onChange={(e) => setForm({ ...form, annual_inc: parseInt(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Debt to Income (DTI) */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-slate-300">Debt-to-Income (DTI %):</label>
                <span className="font-mono text-amber-400 font-bold tabular-nums">
                  {form.dti}%
                </span>
              </div>
              <input
                type="range"
                min="4"
                max="45"
                step="0.5"
                value={form.dti}
                onChange={(e) => setForm({ ...form, dti: parseFloat(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Revolving Utilization */}
            <div className="space-y-1.5">
              <div className="flex justify-between">
                <label className="text-slate-300">Revolving Line Utilization (%):</label>
                <span className="font-mono text-slate-300 font-bold tabular-nums">
                  {form.revol_util}%
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="99"
                step="1"
                value={form.revol_util}
                onChange={(e) => setForm({ ...form, revol_util: parseFloat(e.target.value) })}
                className="w-full accent-indigo-500 cursor-pointer"
              />
            </div>

            {/* Delinquencies & Inquiries */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-400 block mb-1">Delinquencies (24m):</label>
                <input
                  type="number"
                  min="0"
                  max="5"
                  value={form.delinq_2yrs}
                  onChange={(e) =>
                    setForm({ ...form, delinq_2yrs: Math.max(0, parseInt(e.target.value) || 0) })
                  }
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200 font-mono"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Hard Inquiries (6m):</label>
                <input
                  type="number"
                  min="0"
                  max="5"
                  value={form.inq_last_6mths}
                  onChange={(e) =>
                    setForm({ ...form, inq_last_6mths: Math.max(0, parseInt(e.target.value) || 0) })
                  }
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200 font-mono"
                />
              </div>
            </div>

            {/* Home Ownership & Employment */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-400 block mb-1">Home Ownership:</label>
                <select
                  value={form.home_ownership}
                  onChange={(e) =>
                    setForm({ ...form, home_ownership: e.target.value as HomeOwnership })
                  }
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                >
                  <option value="MORTGAGE">Mortgage</option>
                  <option value="RENT">Rent</option>
                  <option value="OWN">Own Outright</option>
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">Employment Length:</label>
                <select
                  value={form.emp_length_years}
                  onChange={(e) =>
                    setForm({ ...form, emp_length_years: parseInt(e.target.value) })
                  }
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                >
                  <option value={0}>Under 1 Year</option>
                  <option value={2}>2-3 Years</option>
                  <option value={5}>4-6 Years</option>
                  <option value={10}>10+ Years</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Real-time Scoring & Decision Console */}
        <div className="lg:col-span-7 space-y-5">
          {/* Main Decision Banner */}
          <div
            className={`p-5 rounded-xl border ${
              isDeclined
                ? 'bg-rose-950/20 border-rose-800/60'
                : isApproved
                ? 'bg-emerald-950/20 border-emerald-800/60'
                : 'bg-amber-950/20 border-amber-800/60'
            }`}
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <div className="text-xs uppercase font-semibold tracking-wider text-slate-400">
                  Automated Underwriting Recommendation
                </div>
                <div
                  className={`text-2xl font-bold mt-1 ${
                    isDeclined
                      ? 'text-rose-400'
                      : isApproved
                      ? 'text-emerald-400'
                      : 'text-amber-400'
                  }`}
                >
                  {prediction.recommendation}
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  Assigned Risk Tier: <span className="font-bold text-white">Tier {prediction.riskTier}</span> ·{' '}
                  Suggested APR: <span className="font-mono text-white font-bold">{prediction.recommendedRate.toFixed(2)}%</span>
                </div>
              </div>

              {/* Gauge Metric */}
              <div className="flex items-center gap-4 bg-slate-950/60 p-3.5 rounded-lg border border-slate-800 self-start sm:self-auto">
                <div className="text-right">
                  <div className="text-3xs text-slate-400 uppercase">Default Probability</div>
                  <div
                    className={`text-2xl font-bold font-mono tabular-nums ${
                      isDeclined
                        ? 'text-rose-400'
                        : isApproved
                        ? 'text-emerald-400'
                        : 'text-amber-400'
                    }`}
                  >
                    {(prediction.defaultProbability * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="w-12 h-12 rounded-full border-2 border-slate-800 flex items-center justify-center font-bold text-sm text-slate-300">
                  {prediction.riskTier}
                </div>
              </div>
            </div>

            {/* Expected Loss & Summary strip */}
            <div className="grid grid-cols-3 gap-3 pt-4 mt-4 border-t border-slate-800/60 text-xs">
              <div>
                <div className="text-slate-400 text-3xs">Expected Portfolio Loss</div>
                <div className="font-mono text-white font-bold mt-0.5">
                  ${prediction.expectedLoss.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-slate-400 text-3xs">Decision Threshold</div>
                <div className="font-mono text-slate-300 font-bold mt-0.5">
                  {(activeMetrics.hyperparameters.decisionThreshold * 100).toFixed(0)}% Default Risk
                </div>
              </div>
              <div>
                <div className="text-slate-400 text-3xs">Confidence Engine</div>
                <div className="text-slate-300 font-medium mt-0.5 truncate">
                  {activeMetrics.algorithmName}
                </div>
              </div>
            </div>
          </div>

          {/* Explainable AI / Feature Contribution Waterfall */}
          <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Explainable AI (XAI) Feature Attribution
                </h4>
                <p className="text-2xs text-slate-400">
                  Local impact of applicant attributes shifting default probability
                </p>
              </div>
              <span className="text-3xs text-slate-500 font-mono">Relative Delta</span>
            </div>

            <div className="space-y-2 pt-1">
              {prediction.featureContributions.map((fc) => {
                const isProtective = fc.contribution < 0;
                return (
                  <div key={fc.feature} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-slate-200">{fc.label}</span>
                        <span className="text-slate-500 text-3xs font-mono">({fc.valueDisplay})</span>
                      </div>
                      <span
                        className={`font-mono tabular-nums font-semibold ${
                          isProtective ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {fc.contribution > 0 ? `+${fc.contribution}%` : `${fc.contribution}%`}
                      </span>
                    </div>
                    {/* Directional Bar */}
                    <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          isProtective ? 'bg-emerald-500' : 'bg-rose-500'
                        }`}
                        style={{
                          width: `${Math.min(100, Math.abs(fc.contribution) * 5)}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* What-If Scenario Sandbox */}
          <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Counterfactual "What-If" Sensitivity Simulator
                </h4>
                <p className="text-2xs text-slate-400">
                  Simulate credit remediation actions and evaluate immediate risk shifts
                </p>
              </div>
              {whatIfScenario && (
                <button
                  onClick={() => setWhatIfScenario(null)}
                  className="text-xs text-indigo-400 hover:text-indigo-300"
                >
                  Reset What-If
                </button>
              )}
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
              <button
                onClick={() =>
                  setWhatIfScenario({
                    label: '+30 FICO Credit Score',
                    patch: { fico_score: Math.min(850, form.fico_score + 30) },
                  })
                }
                className={`p-2.5 rounded-lg border text-left transition-colors ${
                  whatIfScenario?.label === '+30 FICO Credit Score'
                    ? 'bg-indigo-950 border-indigo-500 text-white'
                    : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="font-semibold">+30 FICO Score</div>
                <div className="text-3xs text-slate-500 mt-0.5">Credit boost simulation</div>
              </button>

              <button
                onClick={() =>
                  setWhatIfScenario({
                    label: '-$5,000 Loan Principal',
                    patch: { loan_amnt: Math.max(1000, form.loan_amnt - 5000) },
                  })
                }
                className={`p-2.5 rounded-lg border text-left transition-colors ${
                  whatIfScenario?.label === '-$5,000 Loan Principal'
                    ? 'bg-indigo-950 border-indigo-500 text-white'
                    : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="font-semibold">-$5k Loan Amount</div>
                <div className="text-3xs text-slate-500 mt-0.5">Lower principal demand</div>
              </button>

              <button
                onClick={() =>
                  setWhatIfScenario({
                    label: 'Reduce DTI to 16%',
                    patch: { dti: 16.0 },
                  })
                }
                className={`p-2.5 rounded-lg border text-left transition-colors ${
                  whatIfScenario?.label === 'Reduce DTI to 16%'
                    ? 'bg-indigo-950 border-indigo-500 text-white'
                    : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="font-semibold">Pay Down Debt (DTI 16%)</div>
                <div className="text-3xs text-slate-500 mt-0.5">De-leverage applicant</div>
              </button>
            </div>

            {whatIfScenario && baselinePrediction && (
              <div className="p-3 bg-indigo-950/40 border border-indigo-800/60 rounded-lg flex items-center justify-between text-xs">
                <div>
                  <span className="text-slate-400">Active Scenario: </span>
                  <span className="font-semibold text-white">{whatIfScenario.label}</span>
                </div>
                <div className="font-mono tabular-nums">
                  <span className="text-slate-400">
                    {(baselinePrediction.defaultProbability * 100).toFixed(1)}% →{' '}
                  </span>
                  <span className="font-bold text-emerald-400">
                    {(prediction.defaultProbability * 100).toFixed(1)}%
                  </span>{' '}
                  <span className="text-emerald-400 text-2xs font-semibold">
                    (
                    {(
                      (prediction.defaultProbability - baselinePrediction.defaultProbability) *
                      100
                    ).toFixed(1)}
                    % default risk)
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Underwriting Memo Action */}
          <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Credit Committee Underwriting Memorandum
                </h4>
                <p className="text-2xs text-slate-400">
                  Synthesize an executive underwriting memo with covenants and conditions
                </p>
              </div>
              <button
                onClick={generateUnderwritingMemo}
                disabled={isGeneratingMemo}
                className="px-3.5 py-1.5 text-xs font-semibold text-white bg-indigo-600 rounded-lg hover:bg-indigo-500 transition-colors disabled:opacity-50"
              >
                {isGeneratingMemo ? 'Drafting Memo...' : 'Generate Memo'}
              </button>
            </div>

            {aiMemo && (
              <pre className="p-4 bg-slate-950 border border-slate-800 rounded-lg font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed overflow-x-auto">
                {aiMemo}
              </pre>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
