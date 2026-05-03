import { motion } from "framer-motion";
import { AlertTriangle, AlertCircle, Info, Star } from "lucide-react";
import { ScoreGauge } from "./ScoreGauge";

function formatFlag(flag) {
  return flag.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

export function ReviewCard({ data, delay = 0 }) {
  const {
    is_fake,
    fake_score,
    confidence,
    sentiment,
    sentiment_score,
    sentiment_mismatch,
    behavioral_flags = [],
    nlp_features = {},
    review_text,
    rating
  } = data;

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="relative overflow-hidden rounded-xl border border-white/10 bg-slate-900/50 p-6 shadow-2xl backdrop-blur-sm"
    >
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-600 opacity-50" />
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 space-y-4">
          <div className="flex items-center gap-2 mb-2">
            {rating && (
              <div className="flex items-center gap-1 text-amber-400">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Star key={i} className={`h-4 w-4 ${i < rating ? "fill-current" : "opacity-30"}`} />
                ))}
              </div>
            )}
            <span className="text-xs text-slate-400 font-medium px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
              ID: {data.reviewer_id || "Anonymous"}
            </span>
          </div>
          
          <p className="text-slate-200 text-sm leading-relaxed italic border-l-2 border-slate-700 pl-4 py-1">
            "{review_text || "No review text provided."}"
          </p>

          {sentiment_mismatch && (
            <div className="flex items-center gap-2 mt-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm">
              <AlertTriangle className="h-5 w-5 shrink-0" />
              <span><strong>Warning:</strong> Sentiment mismatch detected. The star rating conflicts with the textual sentiment.</span>
            </div>
          )}

          <div className="space-y-3 pt-4">
            <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-2">
              <AlertCircle className="h-3 w-3" />
              Behavioral Flags
            </h4>
            <div className="flex flex-wrap gap-2">
              {behavioral_flags.length > 0 ? behavioral_flags.map((flag, idx) => (
                <span key={idx} className="px-2.5 py-1 rounded-md text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  {formatFlag(flag)}
                </span>
              )) : (
                <span className="text-xs text-slate-500 italic">No suspicious behavior flagged.</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex flex-col border-l border-white/5 pl-6">
          <ScoreGauge score={fake_score || 0} />
          
          <div className="mt-6 pt-4 border-t border-white/5 space-y-3">
            <h4 className="text-xs uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-2">
              <Info className="h-3 w-3" />
              NLP Features
            </h4>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Words</span>
                <span className="font-mono text-slate-200">{nlp_features.word_count || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Caps</span>
                <span className="font-mono text-slate-200">{Math.round((nlp_features.caps_ratio || 0) * 100)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Excl.</span>
                <span className="font-mono text-slate-200">{nlp_features.exclamation_count || 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Diversity</span>
                <span className="font-mono text-slate-200">{((nlp_features.lexical_diversity || 0) * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
