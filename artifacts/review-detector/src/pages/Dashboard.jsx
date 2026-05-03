import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { Loader2, LayoutDashboard, Activity, ShieldAlert, BarChart3, AlertTriangle } from "lucide-react";
import { Navigation } from "@/components/Navigation";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch("/ml/stats");
        if (!res.ok) throw new Error("Failed to load statistics");
        const data = await res.json();
        setStats(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStats();
  }, []);

  const formatFlag = (flag) => {
    return flag.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  };

  const getSentimentData = () => {
    if (!stats?.sentiment_breakdown) return [];
    return [
      { name: "Positive", value: stats.sentiment_breakdown.positive || 0, color: "#22c55e" },
      { name: "Neutral", value: stats.sentiment_breakdown.neutral || 0, color: "#64748b" },
      { name: "Negative", value: stats.sentiment_breakdown.negative || 0, color: "#ef4444" }
    ];
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <Navigation />
      
      <main className="container mx-auto px-4 py-12 md:py-24">
        <div className="mb-12">
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <LayoutDashboard className="h-8 w-8 text-blue-500" />
            Intelligence Dashboard
          </h1>
          <p className="mt-2 text-slate-400">
            Aggregate statistics and threat intelligence across all analyzed reviews.
          </p>
        </div>

        {loading ? (
          <div className="flex flex-col items-center justify-center py-24 text-slate-400">
            <Loader2 className="h-8 w-8 animate-spin mb-4 text-blue-500" />
            <p>Aggregating intelligence data...</p>
          </div>
        ) : error ? (
          <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400">
            {error}
          </div>
        ) : stats ? (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-slate-900/50 border border-white/5 rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-blue-500"></div>
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Total Analyzed</p>
                    <h3 className="text-3xl font-bold text-white mt-2">{stats.total_analyzed}</h3>
                  </div>
                  <Activity className="h-5 w-5 text-blue-500 opacity-50" />
                </div>
              </div>

              <div className="bg-slate-900/50 border border-white/5 rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-red-500"></div>
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Fake Detected</p>
                    <h3 className="text-3xl font-bold text-white mt-2">{stats.fake_count}</h3>
                  </div>
                  <ShieldAlert className="h-5 w-5 text-red-500 opacity-50" />
                </div>
              </div>

              <div className="bg-slate-900/50 border border-white/5 rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500"></div>
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Fake Rate</p>
                    <h3 className="text-3xl font-bold text-white mt-2">{(stats.fake_rate * 100).toFixed(1)}%</h3>
                  </div>
                  <BarChart3 className="h-5 w-5 text-indigo-500 opacity-50" />
                </div>
              </div>

              <div className="bg-slate-900/50 border border-white/5 rounded-xl p-6 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-1 h-full bg-amber-500"></div>
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Avg Fake Score</p>
                    <h3 className="text-3xl font-bold text-white mt-2">{Math.round(stats.avg_fake_score * 100)}</h3>
                  </div>
                  <AlertTriangle className="h-5 w-5 text-amber-500 opacity-50" />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-slate-900/50 border border-white/5 rounded-xl p-6">
                <h3 className="text-sm uppercase tracking-wider text-slate-400 font-semibold mb-6">Sentiment Distribution</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={getSentimentData()} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                      <XAxis dataKey="name" stroke="#94a3b8" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                      <YAxis stroke="#94a3b8" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                      <Tooltip 
                        cursor={{fill: 'rgba(255,255,255,0.05)'}}
                        contentStyle={{backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc', borderRadius: '0.5rem'}}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {getSentimentData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-900/50 border border-white/5 rounded-xl p-6">
                <h3 className="text-sm uppercase tracking-wider text-slate-400 font-semibold mb-6">Top Behavioral Flags</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={stats.top_flags?.slice(0, 5) || []} margin={{ top: 0, right: 20, left: 20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                      <XAxis type="number" stroke="#94a3b8" tick={{fontSize: 12}} tickLine={false} axisLine={false} />
                      <YAxis type="category" dataKey="flag" stroke="#94a3b8" width={150} tickFormatter={(tick) => formatFlag(tick)} tick={{fontSize: 11}} tickLine={false} axisLine={false} />
                      <Tooltip 
                        cursor={{fill: 'rgba(255,255,255,0.05)'}}
                        contentStyle={{backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc', borderRadius: '0.5rem'}}
                        formatter={(value) => [value, "Count"]}
                        labelFormatter={(label) => formatFlag(label)}
                      />
                      <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={24} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}
