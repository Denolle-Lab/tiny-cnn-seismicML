import { useState, useEffect } from 'react';

export function useWaveformData() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/waveforms.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('Failed to load'))))
      .then(setData)
      .catch(setError);
  }, []);

  return { data, error };
}
