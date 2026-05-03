import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, ShieldAlert, ShieldCheck, Loader2 } from "lucide-react";
import { Navigation } from "@/components/Navigation";
import { ReviewCard } from "@/components/ReviewCard";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [formData, setFormData] = useState({
    review_text: "",
    rating: "",
    reviewer_id: "",
    product: ""
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const analyzeReview = async (e) => {
    e.preventDefault();
    if (!formData.review_text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/ml/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review_text: formData.review_text,
          rating: formData.rating ? parseInt(formData.rating, 10) : undefined,
          reviewer_id: formData.reviewer_id || undefined,
          product: formData.product || undefined
        })
      });

      if (!res.ok) {
        throw new Error("Failed to analyze review");
      }

      const data = await res.json();
      setResult({ ...data, review_text: formData.review_text, rating: formData.rating ? parseInt(formData.rating, 10) : undefined, reviewer_id: formData.reviewer_id });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Navigation />
      
      <main className="container mx-auto px-4 py-12 md:py-24">
        <div className="max-w-4xl mx-auto space-y-12">
          
          <div className="text-center space-y-4">
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white flex justify-center items-center gap-3">
              <ShieldAlert className="h-10 w-10 text-blue-500" />
              Verity Engine
            </h1>
            <p className="text-lg text-slate-400 max-w-2xl mx-auto">
              Advanced ML forensics for e-commerce integrity. Paste a suspicious review below to extract behavioral anomalies, NLP signals, and calculate a precise fake-probability score.
            </p>
          </div>

          <motion.form 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 md:p-8 backdrop-blur-sm shadow-xl"
            onSubmit={analyzeReview}
          >
            <div className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="review_text" className="text-slate-300 font-semibold tracking-wide">Review Content</Label>
                <Textarea 
                  id="review_text"
                  name="review_text"
                  placeholder="Paste the review text here..."
                  className="min-h-[150px] bg-slate-950/50 border-slate-700 text-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                  value={formData.review_text}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="rating" className="text-slate-400 text-xs uppercase tracking-wider">Star Rating (1-5)</Label>
                  <Input 
                    id="rating"
                    name="rating"
                    type="number"
                    min="1"
                    max="5"
                    placeholder="Optional"
                    className="bg-slate-950/50 border-slate-800 focus:border-blue-500"
                    value={formData.rating}
                    onChange={handleChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="reviewer_id" className="text-slate-400 text-xs uppercase tracking-wider">Reviewer ID</Label>
                  <Input 
                    id="reviewer_id"
                    name="reviewer_id"
                    type="text"
                    placeholder="Optional"
                    className="bg-slate-950/50 border-slate-800 focus:border-blue-500"
                    value={formData.reviewer_id}
                    onChange={handleChange}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="product" className="text-slate-400 text-xs uppercase tracking-wider">Product Name</Label>
                  <Input 
                    id="product"
                    name="product"
                    type="text"
                    placeholder="Optional"
                    className="bg-slate-950/50 border-slate-800 focus:border-blue-500"
                    value={formData.product}
                    onChange={handleChange}
                  />
                </div>
              </div>

              <div className="pt-4 flex justify-end">
                <Button 
                  type="submit" 
                  disabled={loading || !formData.review_text.trim()}
                  className="bg-blue-600 hover:bg-blue-500 text-white font-bold tracking-wide px-8"
                >
                  {loading ? (
                    <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> Analyzing...</>
                  ) : (
                    <><Search className="mr-2 h-4 w-4" /> Run Forensics</>
                  )}
                </Button>
              </div>
            </div>
          </motion.form>

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 flex items-center gap-3">
              <ShieldAlert className="h-5 w-5" />
              {error}
            </div>
          )}

          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="py-8">
                  <h3 className="text-xl font-bold text-white mb-6 border-b border-slate-800 pb-4">Analysis Results</h3>
                  <ReviewCard data={result} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
      </main>
    </div>
  );
}
