import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import EDAPage         from "./pages/EDAPage";
import EvaluationPage  from "./pages/EvaluationPage";
import RetrainingPage  from "./pages/RetrainingPage";
import TrainingPage    from "./pages/TrainingPage";

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/"            element={<EDAPage />} />
          <Route path="/training"    element={<TrainingPage />} />
          <Route path="/evaluation"  element={<EvaluationPage />} />
          <Route path="/retraining"  element={<RetrainingPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
