import { motion } from "framer-motion";

export function ScoreGauge({ score }) {
  const percentage = Math.round(score * 100);
  
  let colorClass = "text-green-500";
  let bgClass = "stroke-green-500/20";
  let strokeClass = "stroke-green-500";
  let label = "Genuine";

  if (percentage >= 40 && percentage <= 60) {
    colorClass = "text-amber-500";
    bgClass = "stroke-amber-500/20";
    strokeClass = "stroke-amber-500";
    label = "Uncertain";
  } else if (percentage > 60) {
    colorClass = "text-red-500";
    bgClass = "stroke-red-500/20";
    strokeClass = "stroke-red-500";
    label = "Fake";
  }

  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="flex flex-col items-center justify-center p-4">
      <div className="relative flex items-center justify-center">
        <svg className="w-32 h-32 transform -rotate-90">
          <circle
            className={bgClass}
            strokeWidth="8"
            fill="transparent"
            r={radius}
            cx="64"
            cy="64"
          />
          <motion.circle
            className={`${strokeClass} drop-shadow-md`}
            strokeWidth="8"
            strokeLinecap="round"
            fill="transparent"
            r={radius}
            cx="64"
            cy="64"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            style={{ strokeDasharray: circumference }}
          />
        </svg>
        <div className="absolute flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold ${colorClass}`}>{percentage}%</span>
          <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold mt-1">Score</span>
        </div>
      </div>
      <div className={`mt-4 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-widest border border-current ${colorClass}`}>
        {label}
      </div>
    </div>
  );
}
