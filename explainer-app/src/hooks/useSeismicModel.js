import { useState, useEffect, useCallback } from 'react';
import {
  buildCompactModel,
  loadWeightsFromJson,
  predict as runPredict,
  getActivationsAtSteps as getActivations,
} from '../lib/seismicModel';

const WEIGHTS_URL = '/models/compact_weights.json';

/**
 * Load the compact CNN and expose predict. Class names follow num_classes in weights (2 = Noise, Earthquake).
 */
export function useSeismicModel() {
  const [model, setModel] = useState(null);
  const [classNames, setClassNames] = useState(['Noise', 'Earthquake']);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(WEIGHTS_URL);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const weightsJson = await res.json();
        const numClasses = weightsJson.num_classes ?? 2;
        const names =
          numClasses === 2
            ? ['Noise', 'Earthquake']
            : ['Noise', 'Traffic', 'Earthquake'];
        if (cancelled) return;

        const compact = buildCompactModel(numClasses);
        loadWeightsFromJson(compact, weightsJson);
        if (cancelled) return;

        setModel(compact);
        setClassNames(names);
        setReady(true);
        setError(null);
      } catch (e) {
        if (!cancelled) {
          setError(e?.message ?? 'Failed to load model');
          setReady(false);
        }
      }
    })();

    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    return () => {
      if (model) model.dispose();
    };
  }, [model]);

  const predict = useCallback(
    (waveform) => {
      if (!model || !waveform?.length) return null;
      return runPredict(model, waveform, classNames);
    },
    [model, classNames]
  );

  const getActivationsAtSteps = useCallback(
    (waveform) => {
      if (!model || !waveform?.length) return null;
      return getActivations(model, waveform);
    },
    [model]
  );

  return { model, classNames, ready, error, predict, getActivationsAtSteps };
}
