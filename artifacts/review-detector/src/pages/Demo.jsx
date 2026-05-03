import { useState, useEffect } from "react";
import { Loader2, Database } from "lucide-react";
import { Navigation } from "@/components/Navigation";
import { ReviewCard } from "@/components/ReviewCard";

export default function Demo() {
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDemoData = async () => {
      try {
        const res = await fetch("/ml/demo");
        if (!res.ok) throw new Error("Failed to load demo data");
        const data = await res.json();
        setReviews(data.reviews || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchDemoData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Navigation />
      
      <main className="container mx-auto px-4 py-12 md:py-24">
        <div className="mb-12">
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Database className="h-8 w-8 text-blue-500" />
            Demo Laboratory
          </h1>
          <p className="mt-2 text-slate-400">
            A selection of pre-analyzed reviews to demonstrate Verity Engine capabilities across different behavioral patterns and text structures.
          </p>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-24 text-slate-400">
            <Loader2 className="h-8 w-8 animate-spin mb-4 text-blue-500" />
            <p>Loading demo datasets...</p>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400">
            {error}
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {reviews.map((review, idx) => (
              <ReviewCard key={idx} data={review} delay={idx * 0.1} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
