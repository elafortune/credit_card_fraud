import axios from "axios";

// En dev : proxy Vite → localhost:8001/api
// En prod : VITE_API_URL pointe vers le backend Render
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? "/api",
});

// EDA
export const uploadDataset  = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/eda/upload", form);
};
export const getEdaReport   = () => api.get("/eda/report");

// Training
export const startTraining  = () => api.post("/training/start");
export const getTrainingStatus  = () => api.get("/training/status");
export const getTrainingResults = () => api.get("/training/results");

// Evaluation
export const evaluateDataset = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/evaluation/evaluate-dataset", form);
};
export const getCurrentModel = () => api.get("/evaluation/current-model");

// Retraining
export const detectDrift = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/retraining/detect-drift", form);
};
export const retrainModel = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/retraining/retrain", form);
};
export const getRetrainingStatus = () => api.get("/retraining/status");
