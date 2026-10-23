import React, { useState } from 'react';
import { ModelAlgorithm, ModelHyperparameters, ModelMetrics } from '../types';
import { MLModel, createModel } from '../ml/algorithms';
import { evaluateModel } from '../ml/evaluator';
import { DatasetSplit } from '../ml/featureEngineering';

interface ModelStudioProps {
  split: DatasetSplit;
  metricsMap: Record<ModelAlgorithm, ModelMetrics>;
  modelsMap: Record<ModelAlgorithm, MLModel>;
  activeModelKey: ModelAlgorithm;
  setActiveModelKey: (key: ModelAlgorithm) => void;
  onModelRetrained: (key: ModelAlgorithm, newMetrics: ModelMetrics, newModel: MLModel) => void;
}

export const ModelStudio: React.FC<ModelStudioProps> = ({
  split,
  metricsMap,
  modelsMap,
  activeModelKey,
  setActiveModelKey,
  onModelRetrained,
}) => {
  const activeMetrics = metricsMap[activeModelKey];

  // Hyperparameter Tuning State for the currently selected model
  const [params, setParams] = useState<ModelHyperparameters>(activeMetrics.hyperparameters);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainFeedback, setRetrainFeedback] = useState<string | null>(null);
  const [showRocCurves, setShowRocCurves] = useState(true);

  // Sync state when activeModelKey changes
  React.useEffect(() => {
    setParams(metricsMap[activeModelKey].hyperparameters);
    setRetrainFeedback(null);
  }, [activeModelKey, metricsMap]);

  const handleRetrain = () => {
    setIsRetraining(true);
    setRetrainFeedback(null);

    // Run training with slight micro-delay for smooth UI settling
    setTimeout(() => {
      const startTime = performance.now();
      const newModel = createModel(activeModelKey, params);
      newModel.fit(split.trainX, split.trainY);
      const elapsed = Math.round(performance.now() - startTime);

      const newMetrics = evaluateModel(
        activeModelKey,
        activeMetrics.algorithmName,
        newModel,
        split.testX,
        split.testY,
        elapsed,
        params
      );

      const deltaAuc = (newMetrics.rocAuc - activeMetrics.rocAuc).toFixed(3);
      const sign = Number(deltaAuc) >= 0 ? '+' : '';
      setRetrainFeedback(`Retrained in ${elapsed}ms! ROC-AUC delta: ${sign}${deltaAuc}`);

      onModelRetrained(activeModelKey, newMetrics, newModel);
      setIsRetraining(false);
    }, 120);
  };

  const handleThresholdChange = (val: number) => {
    const updatedParams = { ...params, decisionThreshold: val };
    setParams(updatedParams);

    const updatedMetrics = evaluateModel(
      activeModelKey,
      activeMetrics.algorithmName,
      modelsMap[activeModelKey],
      split.testX,
      split.testY,
      activeMetrics.trainingTimeMs,
      updatedParams
    );

    onModelRetrained(activeModelKey, updatedMetrics, modelsMap[activeModelKey]);
  };

  const algorithmsList: { key: ModelAlgorithm; name: string }[] = [
    { key: 'gradient_boosting', name: 'Gradient Boosted Trees (GBDT)' },
    { key: 'random_forest', name: 'Random Forest Ensemble' },
    { key: 'logistic_regression', name: 'Logistic Regression (L2 Regularized)' },
    { key: 'decision_tree', name: 'Decision Tree (CART)' },
    { key: 'k_nearest_neighbors', name: 'K-Nearest Neighbors (KNN)' },
  ];

  // Best ROC-AUC algorithm
  const bestAlgo = Object.values(metricsMap).reduce((best, curr) =>
    curr.rocAuc > best.rocAuc ? curr : best
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">
            Machine Learning Model Comparison & Evaluation
          </h2>
          <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
            <span>Evaluated on 20% Stratified Test Split (240 loans)</span>
            <span aria-hidden="true">·</span>
            <span>Ground Truth Target: Default (1 = Charged Off)</span>
            <span aria-hidden="true">·</span>
            <span className="text-indigo-400 font-medium">Top Performer: {bestAlgo.algorithmName}</span>
          </div>
        </div>
      </div>

      {/* Model Leaderboard Table */}
      <div className="p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-white">Algorithm Benchmark Leaderboard</h3>
          <span className="text-2xs text-slate-400">Click any row to inspect & tune hyperparameters</span>
        </div>

        <div className="overflow-x-auto rounded-lg border border-slate-800">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-950 text-slate-400 font-medium border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                <th className="py-2.5 px-3 text-right">PR-AUC</th>
                <th className="py-2.5 px-3 text-right">F1-Score</th>
                <th className="py-2.5 px-3 text-right">Accuracy</th>
                <th className="py-2.5 px-3 text-right">Precision</th>
                <th className="py-2.5 px-3 text-right">Recall</th>
                <th className="py-2.5 px-3 text-right">Log-Loss</th>
                <th className="py-2.5 px-3 text-right">Train Time</th>
                <th className="py-2.5 px-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {algorithmsList.map(({ key, name }) => {
                const m = metricsMap[key];
                const isActive = activeModelKey === key;
                const isBest = bestAlgo.algorithm === key;

                return (
                  <tr
                    key={key}
                    onClick={() => setActiveModelKey(key)}
                    className={`cursor-pointer transition-colors ${
                      isActive
                        ? 'bg-indigo-950/40 border-l-2 border-indigo-500'
                        : 'hover:bg-slate-800/40'
                    }`}
                  >
                    <td className="py-3 px-3">
                      <div className="font-semibold text-white flex items-center gap-2">
                        <span>{name}</span>
                        {isBest && (
                          <span className="text-3xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/50 px-1.5 py-0.5 rounded">
                            BEST AUC
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums font-bold text-indigo-400">
                      {m.rocAuc.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-200">
                      {m.prAuc.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-emerald-400 font-semibold">
                      {m.f1Score.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-300">
                      {(m.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-300">
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-300">
                      {(m.recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-400">
                      {m.logLoss.toFixed(3)}
                    </td>
                    <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-400">
                      {m.trainingTimeMs}ms
                    </td>
                    <td className="py-3 px-3 text-center">
                      <span
                        className={`text-2xs font-medium px-2 py-0.5 rounded ${
                          isActive
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {isActive ? 'Active Model' : 'Select'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Deep-Dive Grid: ROC Curve + Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ROC Curves Overlaid (7 Cols) */}
        <div className="lg:col-span-7 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Receiver Operating Characteristic (ROC Curves)
              </h3>
              <p className="text-2xs text-slate-400">
                True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity)
              </p>
            </div>
            <div className="flex items-center gap-1 p-0.5 bg-slate-950 rounded border border-slate-800">
              <button
                onClick={() => setShowRocCurves(true)}
                className={`px-2 py-1 text-3xs font-medium rounded ${
                  showRocCurves ? 'bg-slate-800 text-white' : 'text-slate-400'
                }`}
              >
                Overlay All
              </button>
              <button
                onClick={() => setShowRocCurves(false)}
                className={`px-2 py-1 text-3xs font-medium rounded ${
                  !showRocCurves ? 'bg-slate-800 text-white' : 'text-slate-400'
                }`}
              >
                Active Only
              </button>
            </div>
          </div>

          {/* SVG ROC Plot */}
          <div className="h-64 w-full">
            <svg className="w-full h-full overflow-visible" viewBox="0 0 450 240">
              {/* Plot area: 40 to 420 x, 20 to 200 y */}
              <rect x="50" y="20" width="370" height="180" fill="#0b0f19" rx="4" />
              <line x1="50" y1="200" x2="420" y2="20" stroke="#334155" strokeDasharray="3 3" />

              {/* Gridlines */}
              <line x1="50" y1="155" x2="420" y2="155" stroke="#1e293b" strokeDasharray="2 2" />
              <line x1="50" y1="110" x2="420" y2="110" stroke="#1e293b" strokeDasharray="2 2" />
              <line x1="50" y1="65" x2="420" y2="65" stroke="#1e293b" strokeDasharray="2 2" />
              <line x1="142" y1="20" x2="142" y2="200" stroke="#1e293b" strokeDasharray="2 2" />
              <line x1="235" y1="20" x2="235" y2="200" stroke="#1e293b" strokeDasharray="2 2" />
              <line x1="327" y1="20" x2="327" y2="200" stroke="#1e293b" strokeDasharray="2 2" />

              {/* Axes */}
              <line x1="50" y1="200" x2="420" y2="200" stroke="#475569" />
              <line x1="50" y1="20" x2="50" y2="200" stroke="#475569" />

              {/* Y Axis Ticks */}
              <text x="42" y="204" textAnchor="end" className="fill-slate-500 text-3xs font-mono">0.0</text>
              <text x="42" y="159" textAnchor="end" className="fill-slate-500 text-3xs font-mono">0.25</text>
              <text x="42" y="114" textAnchor="end" className="fill-slate-500 text-3xs font-mono">0.50</text>
              <text x="42" y="69" textAnchor="end" className="fill-slate-500 text-3xs font-mono">0.75</text>
              <text x="42" y="24" textAnchor="end" className="fill-slate-500 text-3xs font-mono">1.0</text>

              {/* X Axis Ticks */}
              <text x="50" y="216" textAnchor="middle" className="fill-slate-500 text-3xs font-mono">0.0</text>
              <text x="142" y="216" textAnchor="middle" className="fill-slate-500 text-3xs font-mono">0.25</text>
              <text x="235" y="216" textAnchor="middle" className="fill-slate-500 text-3xs font-mono">0.50</text>
              <text x="327" y="216" textAnchor="middle" className="fill-slate-500 text-3xs font-mono">0.75</text>
              <text x="420" y="216" textAnchor="middle" className="fill-slate-500 text-3xs font-mono">1.0</text>

              {/* Labels */}
              <text x="235" y="232" textAnchor="middle" className="fill-slate-400 text-2xs font-sans">
                False Positive Rate (FPR)
              </text>
              <text
                x="-110"
                y="18"
                transform="rotate(-90)"
                textAnchor="middle"
                className="fill-slate-400 text-2xs font-sans"
              >
                True Positive Rate (TPR)
              </text>

              {/* Curves */}
              {Object.entries(metricsMap).map(([key, m]) => {
                if (!showRocCurves && key !== activeModelKey) return null;
                const isActive = key === activeModelKey;
                const pathPoints = m.rocCurve.map((pt) => {
                  const px = 50 + pt.x * 370;
                  const py = 200 - pt.y * 180;
                  return `${px},${py}`;
                }).join(' L ');

                let strokeColor = '#64748b';
                if (key === 'gradient_boosting') strokeColor = '#6366f1'; // Indigo
                else if (key === 'random_forest') strokeColor = '#10b981'; // Emerald
                else if (key === 'logistic_regression') strokeColor = '#f59e0b'; // Amber
                else if (key === 'decision_tree') strokeColor = '#06b6d4'; // Cyan
                else if (key === 'k_nearest_neighbors') strokeColor = '#ec4899'; // Pink

                return (
                  <path
                    key={key}
                    d={`M ${pathPoints}`}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={isActive ? '3' : '1.5'}
                    strokeOpacity={isActive ? '1' : '0.65'}
                  />
                );
              })}
            </svg>
          </div>

          {/* Model Curves Legend */}
          <div className="flex flex-wrap items-center gap-4 text-2xs text-slate-300 pt-2 border-t border-slate-800">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-indigo-500" />
              <span>GBDT ({metricsMap.gradient_boosting.rocAuc.toFixed(3)})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-emerald-500" />
              <span>Random Forest ({metricsMap.random_forest.rocAuc.toFixed(3)})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-amber-500" />
              <span>Logistic Reg ({metricsMap.logistic_regression.rocAuc.toFixed(3)})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-cyan-500" />
              <span>Decision Tree ({metricsMap.decision_tree.rocAuc.toFixed(3)})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-0.5 bg-pink-500" />
              <span>KNN ({metricsMap.k_nearest_neighbors.rocAuc.toFixed(3)})</span>
            </span>
          </div>
        </div>

        {/* Confusion Matrix & Threshold Tuning (5 Cols) */}
        <div className="lg:col-span-5 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Confusion Matrix & Decision Threshold
            </h3>
            <p className="text-2xs text-slate-400">
              Active model: <span className="text-white font-medium">{activeMetrics.algorithmName}</span>
            </p>
          </div>

          {/* Interactive Threshold Slider */}
          <div className="p-3.5 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Cutoff Threshold:</span>
              <span className="font-mono text-indigo-400 font-bold tabular-nums">
                {params.decisionThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={params.decisionThreshold}
              onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-3xs text-slate-500 font-mono">
              <span>0.10 (High Recall / Strict)</span>
              <span>0.50 (Standard)</span>
              <span>0.90 (High Precision)</span>
            </div>
          </div>

          {/* 2x2 Matrix */}
          <div className="grid grid-cols-2 gap-3 text-center">
            {/* True Negative */}
            <div className="p-3 bg-emerald-950/30 border border-emerald-800/40 rounded-lg">
              <div className="text-3xs text-emerald-400 uppercase font-semibold">
                True Negatives (TN)
              </div>
              <div className="text-2xl font-bold font-mono text-emerald-300 mt-1">
                {activeMetrics.confusionMatrix.tn}
              </div>
              <div className="text-3xs text-slate-400 mt-0.5">
                Correctly approved good loans
              </div>
            </div>

            {/* False Positive */}
            <div className="p-3 bg-rose-950/30 border border-rose-800/40 rounded-lg">
              <div className="text-3xs text-rose-400 uppercase font-semibold">
                False Positives (FP)
              </div>
              <div className="text-2xl font-bold font-mono text-rose-300 mt-1">
                {activeMetrics.confusionMatrix.fp}
              </div>
              <div className="text-3xs text-slate-400 mt-0.5">
                Good loans incorrectly rejected
              </div>
            </div>

            {/* False Negative */}
            <div className="p-3 bg-amber-950/30 border border-amber-800/40 rounded-lg">
              <div className="text-3xs text-amber-400 uppercase font-semibold">
                False Negatives (FN)
              </div>
              <div className="text-2xl font-bold font-mono text-amber-300 mt-1">
                {activeMetrics.confusionMatrix.fn}
              </div>
              <div className="text-3xs text-slate-400 mt-0.5">
                Defaults missed (Credit Loss!)
              </div>
            </div>

            {/* True Positive */}
            <div className="p-3 bg-indigo-950/30 border border-indigo-800/40 rounded-lg">
              <div className="text-3xs text-indigo-400 uppercase font-semibold">
                True Positives (TP)
              </div>
              <div className="text-2xl font-bold font-mono text-indigo-300 mt-1">
                {activeMetrics.confusionMatrix.tp}
              </div>
              <div className="text-3xs text-slate-400 mt-0.5">
                Defaults correctly intercepted
              </div>
            </div>
          </div>

          {/* Quick Metrics Badges */}
          <div className="grid grid-cols-2 gap-2 pt-1 text-xs">
            <div className="p-2 bg-slate-950 border border-slate-800 rounded flex justify-between">
              <span className="text-slate-400">Sensitivity (Recall):</span>
              <span className="font-mono text-white font-semibold">
                {(activeMetrics.recall * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-2 bg-slate-950 border border-slate-800 rounded flex justify-between">
              <span className="text-slate-400">Specificity:</span>
              <span className="font-mono text-white font-semibold">
                {(activeMetrics.specificity * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Feature Importance & Hyperparameter Tuning Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Feature Importance Ranking (6 Cols) */}
        <div className="lg:col-span-6 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Feature Importance ({activeMetrics.algorithmName})
            </h3>
            <p className="text-2xs text-slate-400">
              Relative predictive weight assigned by the model in determining default
            </p>
          </div>

          <div className="space-y-2.5">
            {activeMetrics.featureImportances.slice(0, 8).map((fi) => (
              <div key={fi.feature} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-200 font-medium">{fi.displayName}</span>
                  <span className="font-mono text-slate-400 tabular-nums">
                    {(fi.importance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                    style={{ width: `${Math.max(4, fi.importance * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Hyperparameter Tuning Studio (6 Cols) */}
        <div className="lg:col-span-6 p-5 bg-slate-900 border border-slate-800 rounded-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-white">
                Hyperparameter Tuning & Live Retraining
              </h3>
              <p className="text-2xs text-slate-400">
                Adjust hyperparameters and execute live model training
              </p>
            </div>
            {retrainFeedback && (
              <span className="text-3xs font-mono text-emerald-400 bg-emerald-950 px-2 py-1 rounded border border-emerald-800">
                {retrainFeedback}
              </span>
            )}
          </div>

          <div className="space-y-3">
            {/* Tree Depth */}
            {(activeModelKey === 'random_forest' ||
              activeModelKey === 'gradient_boosting' ||
              activeModelKey === 'decision_tree') && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">Max Depth:</span>
                  <span className="font-mono text-white font-bold">{params.maxDepth ?? 5}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="10"
                  step="1"
                  value={params.maxDepth ?? 5}
                  onChange={(e) =>
                    setParams({ ...params, maxDepth: parseInt(e.target.value) })
                  }
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>
            )}

            {/* Estimators / Trees */}
            {(activeModelKey === 'random_forest' || activeModelKey === 'gradient_boosting') && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">Number of Estimators (Trees):</span>
                  <span className="font-mono text-white font-bold">{params.nEstimators ?? 25}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="60"
                  step="5"
                  value={params.nEstimators ?? 25}
                  onChange={(e) =>
                    setParams({ ...params, nEstimators: parseInt(e.target.value) })
                  }
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>
            )}

            {/* Learning Rate */}
            {(activeModelKey === 'gradient_boosting' ||
              activeModelKey === 'logistic_regression') && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">Learning Rate (eta):</span>
                  <span className="font-mono text-white font-bold">
                    {(params.learningRate ?? 0.1).toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.02"
                  max="0.40"
                  step="0.02"
                  value={params.learningRate ?? 0.1}
                  onChange={(e) =>
                    setParams({ ...params, learningRate: parseFloat(e.target.value) })
                  }
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>
            )}

            {/* Regularization C */}
            {activeModelKey === 'logistic_regression' && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">L2 Regularization Strength (C):</span>
                  <span className="font-mono text-white font-bold">
                    {(params.regularizationC ?? 1.0).toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="10.0"
                  step="0.5"
                  value={params.regularizationC ?? 1.0}
                  onChange={(e) =>
                    setParams({ ...params, regularizationC: parseFloat(e.target.value) })
                  }
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>
            )}

            {/* KNN K */}
            {activeModelKey === 'k_nearest_neighbors' && (
              <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">Neighbors (k):</span>
                  <span className="font-mono text-white font-bold">{params.kNeighbors ?? 7}</span>
                </div>
                <input
                  type="range"
                  min="3"
                  max="19"
                  step="2"
                  value={params.kNeighbors ?? 7}
                  onChange={(e) =>
                    setParams({ ...params, kNeighbors: parseInt(e.target.value) })
                  }
                  className="w-full accent-indigo-500 cursor-pointer"
                />
              </div>
            )}

            <button
              onClick={handleRetrain}
              disabled={isRetraining}
              className="w-full py-2.5 px-4 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-500 rounded-lg transition-colors flex items-center justify-center gap-2 shadow-sm disabled:opacity-50"
            >
              {isRetraining ? 'Fitting Model...' : `Retrain ${activeMetrics.algorithmName}`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
