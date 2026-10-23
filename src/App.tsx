import React, { useState, useMemo } from 'react';
import { ModelAlgorithm, ModelMetrics } from './types';
import { INITIAL_PROCESSED_DATASET } from './data/loanDataset';
import { createStratifiedSplit } from './ml/featureEngineering';
import { MLModel, createModel } from './ml/algorithms';
import { evaluateModel } from './ml/evaluator';
import { TopNav, ActiveTab } from './components/TopNav';
import { EdaHub } from './components/EdaHub';
import { DataCleaningView } from './components/DataCleaningView';
import { ModelStudio } from './components/ModelStudio';
import { PredictionInterface } from './components/PredictionInterface';
import { BatchProcessor } from './components/BatchProcessor';
import { CodeExport } from './components/CodeExport';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('eda');
  const [activeModelKey, setActiveModelKey] = useState<ModelAlgorithm>('gradient_boosting');

  // Generate stratified dataset split once
  const split = useMemo(() => {
    return createStratifiedSplit(INITIAL_PROCESSED_DATASET, 0.2);
  }, []);

  // Train and evaluate all models on initialization
  const initialModelsAndMetrics = useMemo(() => {
    const algorithms: { key: ModelAlgorithm; name: string }[] = [
      { key: 'gradient_boosting', name: 'Gradient Boosted Trees (GBDT)' },
      { key: 'random_forest', name: 'Random Forest Ensemble' },
      { key: 'logistic_regression', name: 'Logistic Regression (L2 Regularized)' },
      { key: 'decision_tree', name: 'Decision Tree (CART)' },
      { key: 'k_nearest_neighbors', name: 'K-Nearest Neighbors (KNN)' },
    ];

    const models: Partial<Record<ModelAlgorithm, MLModel>> = {};
    const metrics: Partial<Record<ModelAlgorithm, ModelMetrics>> = {};

    algorithms.forEach(({ key, name }) => {
      const defaultParams = {
        learningRate: key === 'gradient_boosting' ? 0.12 : 0.08,
        maxDepth: key === 'decision_tree' ? 5 : key === 'gradient_boosting' ? 3 : 5,
        nEstimators: key === 'gradient_boosting' ? 20 : 25,
        regularizationC: 1.0,
        minSamplesSplit: 10,
        kNeighbors: 7,
        decisionThreshold: 0.5,
      };

      const startTime = performance.now();
      const model = createModel(key, defaultParams);
      model.fit(split.trainX, split.trainY);
      const elapsed = Math.round(performance.now() - startTime);

      const evaluated = evaluateModel(
        key,
        name,
        model,
        split.testX,
        split.testY,
        elapsed,
        defaultParams
      );

      models[key] = model;
      metrics[key] = evaluated;
    });

    return {
      models: models as Record<ModelAlgorithm, MLModel>,
      metrics: metrics as Record<ModelAlgorithm, ModelMetrics>,
    };
  }, [split]);

  const [modelsMap, setModelsMap] = useState<Record<ModelAlgorithm, MLModel>>(
    initialModelsAndMetrics.models
  );
  const [metricsMap, setMetricsMap] = useState<Record<ModelAlgorithm, ModelMetrics>>(
    initialModelsAndMetrics.metrics
  );

  const handleModelRetrained = (
    key: ModelAlgorithm,
    newMetrics: ModelMetrics,
    newModel: MLModel
  ) => {
    setModelsMap((prev) => ({ ...prev, [key]: newModel }));
    setMetricsMap((prev) => ({ ...prev, [key]: newMetrics }));
  };

  const handleQuickPredict = () => {
    setActiveTab('predict');
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-white">
      {/* Strict 3-Zone Top Bar Navigation */}
      <TopNav
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onQuickPredict={handleQuickPredict}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'eda' && <EdaHub dataset={INITIAL_PROCESSED_DATASET} />}
        {activeTab === 'cleaning' && (
          <DataCleaningView dataset={INITIAL_PROCESSED_DATASET} split={split} />
        )}
        {activeTab === 'models' && (
          <ModelStudio
            split={split}
            metricsMap={metricsMap}
            modelsMap={modelsMap}
            activeModelKey={activeModelKey}
            setActiveModelKey={setActiveModelKey}
            onModelRetrained={handleModelRetrained}
          />
        )}
        {activeTab === 'predict' && (
          <PredictionInterface
            split={split}
            activeModel={modelsMap[activeModelKey]}
            activeMetrics={metricsMap[activeModelKey]}
            allModels={modelsMap}
          />
        )}
        {activeTab === 'batch' && (
          <BatchProcessor
            dataset={INITIAL_PROCESSED_DATASET}
            split={split}
            activeModel={modelsMap[activeModelKey]}
            activeMetrics={metricsMap[activeModelKey]}
          />
        )}
        {activeTab === 'code' && (
          <CodeExport
            activeModelKey={activeModelKey}
            activeMetrics={metricsMap[activeModelKey]}
          />
        )}
      </main>

      {/* Quiet Professional Footer */}
      <footer className="mt-auto border-t border-slate-800/80 bg-slate-950 py-6 px-6 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-200">Credence Risk ML</span>
            <span aria-hidden="true">·</span>
            <span>Machine Learning Loan Risk Prediction System</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-slate-500">v1.2.0</span>
          </div>

          <div className="flex items-center gap-6">
            <button
              onClick={() => setActiveTab('eda')}
              className="hover:text-slate-200 transition-colors"
            >
              EDA
            </button>
            <button
              onClick={() => setActiveTab('models')}
              className="hover:text-slate-200 transition-colors"
            >
              Models
            </button>
            <button
              onClick={() => setActiveTab('predict')}
              className="hover:text-slate-200 transition-colors"
            >
              Sandbox
            </button>
            <button
              onClick={() => setActiveTab('code')}
              className="hover:text-slate-200 transition-colors"
            >
              FastAPI Export
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
