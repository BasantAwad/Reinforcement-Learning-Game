import React, { useState, useEffect, useRef } from "react";
import {
  Play,
  Square,
  Activity,
  Zap,
  Grid,
  Ghost,
  User,
  Cpu,
  ScrollText,
  AlertTriangle,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
type Game = "JungleDash";
type Algorithm = "DP" | "Q-Learning" | "SARSA" | "REINFORCE";
type GameState = "idle" | "running" | "stopped";
type EnvMode = "static" | "dynamic";

interface LogEntry {
  message: string;
  logType: "action" | "reward" | "penalty" | "info" | "success" | "failure";
  step: number;
  timestamp: Date;
}

// Available games
const GAMES: Game[] = ["JungleDash"];

// Main App Component
function App() {
  const [isConnected, setIsConnected] = useState(false);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const [selectedGame, setSelectedGame] = useState<Game>("JungleDash");
  const [selectedAlgo, setSelectedAlgo] = useState<Algorithm>("Q-Learning");
  const [envMode, setEnvMode] = useState<EnvMode>("static");
  const [gameState, setGameState] = useState<GameState>("idle");
  const [stats, setStats] = useState<
    {
      reward: number;
      steps: number;
      penalties: number;
      episode: number;
      epsilon?: number;
    }[]
  >([{ reward: 0, steps: 0, penalties: 0, episode: 0 }]);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [totalPenalties, setTotalPenalties] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const logContainerRef = useRef<HTMLDivElement | null>(null);

  const connectWS = () => {
    // Prevent multiple connections
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    )
      return;

    console.log("Attempting WS Connection...");
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
      console.log("WS Connected");
      setIsConnected(true);
      if (reconnectTimeoutRef.current)
        clearTimeout(reconnectTimeoutRef.current);
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "frame") {
        setCurrentFrame(`data:image/jpeg;base64,${msg.data}`);
      } else if (msg.type === "stats") {
        setStats((prev) => [...prev.slice(-99), msg.data]);
        if (msg.data.penalties !== undefined) {
          setTotalPenalties(msg.data.penalties);
        }
      } else if (msg.type === "log") {
        const newLog: LogEntry = {
          message: msg.data.message,
          logType: msg.data.logType,
          step: msg.data.step,
          timestamp: new Date(),
        };
        setLogs((prev) => [...prev.slice(-199), newLog]);
      } else if (msg.type === "info") {
        console.log(`Info: ${msg.message}`);
      } else if (msg.type === "error") {
        console.error(`Backend Error: ${msg.message}`);
        alert(`Error: ${msg.message}`);
      }
    };

    ws.onclose = (event) => {
      console.log(`WS Disconnected (Code: ${event.code})`);
      if (wsRef.current === ws) {
        setIsConnected(false);
        wsRef.current = null;
        setGameState("stopped");
        reconnectTimeoutRef.current = setTimeout(connectWS, 3000);
      } else {
        console.log("Ignoring stale WS close event");
      }
    };

    ws.onerror = (err) => {
      console.log("WS Error occurred");
    };

    wsRef.current = ws;
  };

  useEffect(() => {
    connectWS();
    return () => {
      if (wsRef.current) {
        if (
          wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING
        ) {
          wsRef.current.close();
        }
      }
      if (reconnectTimeoutRef.current)
        clearTimeout(reconnectTimeoutRef.current);
    };
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleStart = () => {
    console.log("Start button clicked!");
    if (isConnected && wsRef.current) {
      console.log("Sending start message...");
      setStats([{ reward: 0, steps: 0, penalties: 0, episode: 0 }]);
      setLogs([]);
      setTotalPenalties(0);
      wsRef.current.send(
        JSON.stringify({
          type: "start",
          gameId: selectedGame,
          algoId: selectedAlgo,
          envMode: envMode,
        })
      );
      console.log("Start message sent!");
      setGameState("running");
    } else {
      console.error("Cannot start: Not connected or WebSocket unavailable");
      alert("Not connected to backend! Check if backend is running.");
    }
  };

  const handleStop = () => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
      setGameState("stopped");
    }
  };

  const handleGenerateReport = async () => {
    try {
      const response = await fetch("http://localhost:8000/api/report/generate");
      const data = await response.json();
      alert(
        `Report generated successfully!\n\n${data.message}\n\nYou can download it from the backend folder.`
      );

      // Optionally download it
      window.open("http://localhost:8000/api/report/download", "_blank");
    } catch (error) {
      console.error("Error generating report:", error);
      alert(
        "Failed to generate report. Make sure you have run some algorithms first."
      );
    }
  };

  const getLogColor = (logType: string) => {
    switch (logType) {
      case "action":
        return "text-slate-300";
      case "reward":
        return "text-emerald-400";
      case "penalty":
        return "text-red-400";
      case "success":
        return "text-green-400 font-bold";
      case "failure":
        return "text-red-500 font-bold";
      case "info":
        return "text-blue-400";
      default:
        return "text-slate-400";
    }
  };

  const getLogIcon = (logType: string) => {
    switch (logType) {
      case "penalty":
        return "⚠️";
      case "reward":
        return "🎁";
      case "success":
        return "🏆";
      case "failure":
        return "💀";
      case "info":
        return "ℹ️";
      default:
        return "→";
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white font-sans selection:bg-purple-500/30">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              RL Arena
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-400">
            <span className="flex items-center gap-1">
              <Cpu size={14} /> Local Backend
            </span>
            <span
              className={cn(
                "w-2 h-2 rounded-full animate-pulse",
                isConnected ? "bg-green-500" : "bg-red-500"
              )}
            ></span>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8 grid grid-cols-12 gap-8">
        {/* Sidebar Controls */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          {/* Game Selection */}
          <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700/50">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Grid size={16} /> Environment
            </h2>
            <div className="space-y-2">
              {GAMES.map((game) => (
                <button
                  key={game}
                  onClick={() => setSelectedGame(game)}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-xl transition-all duration-200 flex items-center justify-between group",
                    selectedGame === game
                      ? "bg-purple-600 text-white shadow-lg shadow-purple-900/20"
                      : "bg-slate-900/50 text-slate-400 hover:bg-slate-800 hover:text-white"
                  )}
                >
                  <span>{game}</span>
                  {selectedGame === game && (
                    <div className="w-2 h-2 bg-white rounded-full" />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Algorithm Selection */}
          <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700/50">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Zap size={16} /> Algorithm
            </h2>
            <div className="space-y-2">
              {(["DP", "Q-Learning", "SARSA", "REINFORCE"] as Algorithm[]).map(
                (algo) => (
                  <button
                    key={algo}
                    onClick={() => setSelectedAlgo(algo)}
                    className={cn(
                      "w-full text-left px-4 py-3 rounded-xl transition-all duration-200 flex items-center justify-between",
                      selectedAlgo === algo
                        ? "bg-blue-600 text-white shadow-lg shadow-blue-900/20"
                        : "bg-slate-900/50 text-slate-400 hover:bg-slate-800 hover:text-white"
                    )}
                  >
                    <span>{algo}</span>
                    {selectedAlgo === algo && (
                      <div className="w-2 h-2 bg-white rounded-full" />
                    )}
                  </button>
                )
              )}
            </div>
          </div>

          {/* Environment Mode Selection */}
          <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700/50">
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Grid size={16} /> Environment Mode
            </h2>
            <div className="space-y-2">
              {(["static", "dynamic"] as EnvMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setEnvMode(mode)}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-xl transition-all duration-200 flex items-center justify-between",
                    envMode === mode
                      ? "bg-cyan-600 text-white shadow-lg shadow-cyan-900/20"
                      : "bg-slate-900/50 text-slate-400 hover:bg-slate-800 hover:text-white"
                  )}
                >
                  <div>
                    <span className="capitalize">{mode}</span>
                    {mode === "static" && (
                      <p className="text-xs opacity-70 mt-0.5">
                        Rewards persist, fixed goal bonus
                      </p>
                    )}
                    {mode === "dynamic" && (
                      <p className="text-xs opacity-70 mt-0.5">
                        Rewards disappear, scaling bonus
                      </p>
                    )}
                  </div>
                  {envMode === mode && (
                    <div className="w-2 h-2 bg-white rounded-full" />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={handleStart}
              disabled={gameState === "running"}
              className="flex items-center justify-center gap-2 py-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold shadow-lg shadow-emerald-900/20"
            >
              <Play size={20} fill="currentColor" /> Start
            </button>
            <button
              onClick={handleStop}
              disabled={gameState !== "running"}
              className="flex items-center justify-center gap-2 py-4 rounded-xl bg-red-600 hover:bg-red-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold shadow-lg shadow-red-900/20"
            >
              <Square size={20} fill="currentColor" /> Stop
            </button>
          </div>

          {/* Generate Report Button */}
          <button
            onClick={handleGenerateReport}
            className="w-full flex items-center justify-center gap-2 py-4 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white transition-all font-semibold shadow-lg shadow-purple-900/20"
          >
            <ScrollText size={20} /> Generate Report
          </button>
        </div>

        {/* Main Viewport */}
        <div className="col-span-12 lg:col-span-9 space-y-6">
          {/* Visualizer */}
          <div className="rounded-3xl overflow-hidden bg-black border border-slate-800 shadow-2xl relative aspect-video flex items-center justify-center group">
            {currentFrame ? (
              <img
                src={currentFrame}
                alt="Game Stream"
                className="w-full h-full object-contain pixelated"
                style={{ imageRendering: "pixelated" }}
              />
            ) : (
              <div className="text-center space-y-4">
                <div className="w-24 h-24 rounded-full bg-slate-900/80 flex items-center justify-center mx-auto border border-slate-700">
                  <Ghost className="w-10 h-10 text-slate-500" />
                </div>
                <div>
                  <p className="text-slate-400 text-lg">
                    Ready to initialize environment
                  </p>
                  <p className="text-slate-600 text-sm">
                    Select a game and algorithm to begin
                  </p>
                </div>
              </div>
            )}

            {/* Overlay Stats */}
            <div className="absolute top-4 right-4 flex gap-3">
              <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/10 text-xs font-mono text-emerald-400">
                Episode:{" "}
                {stats.length > 0 ? stats[stats.length - 1].episode || 0 : 0}
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/10 text-xs font-mono text-blue-400">
                Steps:{" "}
                {stats.length > 0 ? stats[stats.length - 1].steps || 0 : 0}
              </div>
              <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/10 text-xs font-mono text-red-400">
                Penalties: {totalPenalties.toFixed(1)}
              </div>
            </div>
          </div>

          {/* Metrics & Logs Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Rewards Chart */}
            <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 h-80">
              <h3 className="text-slate-200 font-semibold mb-6 flex items-center gap-2">
                <Activity size={18} className="text-blue-400" /> Training
                Rewards
              </h3>
              <ResponsiveContainer width="100%" height="80%">
                <LineChart data={stats}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#334155"
                    vertical={false}
                  />
                  <XAxis dataKey="steps" hide />
                  <YAxis
                    stroke="#94a3b8"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1e293b",
                      borderColor: "#334155",
                      borderRadius: "8px",
                    }}
                    itemStyle={{ color: "#fff" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="reward"
                    stroke="#8b5cf6"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 6, fill: "#8b5cf6" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Agent Status */}
            <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 h-80">
              <h3 className="text-slate-200 font-semibold mb-4 flex items-center gap-2">
                <User size={18} className="text-purple-400" /> Agent Status
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 text-sm">Status</span>
                  <span
                    className={cn(
                      "px-2 py-0.5 rounded text-xs font-bold uppercase",
                      gameState === "running"
                        ? "bg-emerald-500/20 text-emerald-400"
                        : "bg-slate-700/50 text-slate-400"
                    )}
                  >
                    {gameState}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 text-sm">Total Steps</span>
                  <span className="font-mono text-blue-400">
                    {stats.length > 0 ? stats[stats.length - 1].steps || 0 : 0}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                  <span className="text-slate-400 text-sm">Current Reward</span>
                  <span className="font-mono text-purple-400">
                    {stats.length > 0
                      ? (stats[stats.length - 1].reward || 0).toFixed(2)
                      : 0}
                  </span>
                </div>
                {stats.length > 0 &&
                  stats[stats.length - 1].epsilon !== undefined &&
                  stats[stats.length - 1].epsilon !== null && (
                    <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                      <span className="text-slate-400 text-sm">
                        Exploration (ε)
                      </span>
                      <span className="font-mono text-yellow-400">
                        {(stats[stats.length - 1].epsilon! * 100).toFixed(1)}%
                      </span>
                    </div>
                  )}
                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-red-900/30">
                  <span className="text-slate-400 text-sm flex items-center gap-1">
                    <AlertTriangle size={14} className="text-red-400" /> Total
                    Penalties
                  </span>
                  <span className="font-mono text-red-400">
                    {totalPenalties.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Agent Action Logs */}
          <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50">
            <h3 className="text-slate-200 font-semibold mb-4 flex items-center gap-2">
              <ScrollText size={18} className="text-cyan-400" /> Agent Action
              Log
            </h3>
            <div
              ref={logContainerRef}
              className="h-48 overflow-y-auto bg-slate-900/70 rounded-xl p-4 font-mono text-sm space-y-1 border border-slate-800"
            >
              {logs.length === 0 ? (
                <p className="text-slate-500 text-center py-8">
                  Start training to see agent actions...
                </p>
              ) : (
                logs.map((log, idx) => (
                  <div
                    key={idx}
                    className={cn(
                      "flex items-start gap-2",
                      getLogColor(log.logType)
                    )}
                  >
                    <span className="text-slate-600 w-16 shrink-0">
                      [{log.step}]
                    </span>
                    <span className="w-5">{getLogIcon(log.logType)}</span>
                    <span>{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
