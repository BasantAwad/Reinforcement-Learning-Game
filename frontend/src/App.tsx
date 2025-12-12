import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Activity, Settings, Zap, Grid, Ghost, User, Cpu } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

// Types
type Game = 'MsPacman' | 'KungFuMaster' | 'MiniWorld-Maze';
type Algorithm = 'DP' | 'Q-Learning' | 'SARSA' | 'DQN' | 'PG';
type GameState = 'idle' | 'running' | 'stopped';

// Main App Component
function App() {
    const [isConnected, setIsConnected] = useState(false);
    const reconnectTimeoutRef = useRef<number | null>(null);
    const [selectedGame, setSelectedGame] = useState<Game>('MsPacman');
    const [selectedAlgo, setSelectedAlgo] = useState<Algorithm>('DQN');
    const [gameState, setGameState] = useState<GameState>('idle');
    const [stats, setStats] = useState<{ reward: number, steps: number }[]>([{ reward: 0, steps: 0 }]);
    const [currentFrame, setCurrentFrame] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const connectWS = () => {
        // Prevent multiple connections
        if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) return;

        console.log("Attempting WS Connection...");
        const ws = new WebSocket('ws://localhost:8000/ws');

        ws.onopen = () => {
            console.log("WS Connected");
            setIsConnected(true);
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'frame') {
                setCurrentFrame(`data:image/jpeg;base64,${msg.data}`);
            } else if (msg.type === 'stats') {
                setStats(prev => [...prev.slice(-99), msg.data]);
            } else if (msg.type === 'info') {
                console.log(`Info: ${msg.message}`);
            } else if (msg.type === 'error') {
                console.error(`Backend Error: ${msg.message}`);
                alert(`Error: ${msg.message}`); // Make it visible to user
            }
        };

        ws.onclose = (event) => {
            console.log(`WS Disconnected (Code: ${event.code})`);
            // Only handle if THIS is still the current websocket
            if (wsRef.current === ws) {
                setIsConnected(false);
                wsRef.current = null;
                setGameState('stopped');
                // Auto-reconnect
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
                // Only close if open or connecting
                if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
                    wsRef.current.close();
                }
            }
            if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        };
    }, []);

    const handleStart = () => {
        console.log("Start button clicked!");
        console.log("isConnected:", isConnected);
        console.log("wsRef.current:", wsRef.current);
        console.log("wsRef.current.readyState:", wsRef.current?.readyState);

        if (isConnected && wsRef.current) {
            console.log("Sending start message...");
            setStats([]);
            wsRef.current.send(JSON.stringify({
                type: 'start',
                gameId: selectedGame,
                algoId: selectedAlgo
            }));
            console.log("Start message sent!");
            setGameState('running');
        } else {
            console.error("Cannot start: Not connected or WebSocket unavailable");
            alert("Not connected to backend! Check if backend is running.");
        }
    };

    const handleStop = () => {
        if (wsRef.current) {
            wsRef.current.send(JSON.stringify({ type: 'stop' }));
            setGameState('stopped');
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
                        <span className="flex items-center gap-1"><Cpu size={14} /> Local Backend</span>
                        <span className={cn("w-2 h-2 rounded-full animate-pulse", isConnected ? "bg-green-500" : "bg-red-500")}></span>
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
                            {(['MsPacman', 'KungFuMaster', 'MiniWorld-Maze'] as Game[]).map(game => (
                                <button
                                    key={game}
                                    onClick={() => {
                                        setSelectedGame(game);
                                        // Logic: If switching to Atari (Pacman/KungFu), switch to DQN if current is Tabular/DP
                                        const isAtari = game !== 'MiniWorld-Maze';
                                        if (isAtari && ['DP', 'Q-Learning', 'SARSA'].includes(selectedAlgo)) {
                                            setSelectedAlgo('DQN');
                                        }
                                        // If switching to Maze, it supports all (using fallbacks), so no forced switch needed.
                                    }}
                                    className={cn(
                                        "w-full text-left px-4 py-3 rounded-xl transition-all duration-200 flex items-center justify-between group",
                                        selectedGame === game
                                            ? "bg-purple-600 text-white shadow-lg shadow-purple-900/20"
                                            : "bg-slate-900/50 text-slate-400 hover:bg-slate-800 hover:text-white"
                                    )}
                                >
                                    <span>{game}</span>
                                    {selectedGame === game && <div className="w-2 h-2 bg-white rounded-full" />}
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
                            {(['DP', 'Q-Learning', 'SARSA', 'DQN', 'PG'] as Algorithm[]).map(algo => {
                                const isAtari = selectedGame !== 'MiniWorld-Maze';
                                const isDeep = ['DQN', 'PG'].includes(algo);
                                const disabled = isAtari && !isDeep; // Disable Tabular/DP for Atari

                                return (
                                    <button
                                        key={algo}
                                        onClick={() => setSelectedAlgo(algo)}
                                        disabled={disabled}
                                        className={cn(
                                            "w-full text-left px-4 py-3 rounded-xl transition-all duration-200 flex items-center justify-between",
                                            selectedAlgo === algo
                                                ? "bg-blue-600 text-white shadow-lg shadow-blue-900/20"
                                                : "bg-slate-900/50 text-slate-400 hover:bg-slate-800 hover:text-white",
                                            disabled && "opacity-40 cursor-not-allowed bg-slate-900/30 text-slate-600"
                                        )}
                                    >
                                        <span>{algo}</span>
                                        {selectedAlgo === algo && <div className="w-2 h-2 bg-white rounded-full" />}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="grid grid-cols-2 gap-3">
                        <button
                            onClick={handleStart}
                            disabled={gameState === 'running'}
                            className="flex items-center justify-center gap-2 py-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold shadow-lg shadow-emerald-900/20"
                        >
                            <Play size={20} fill="currentColor" /> Start
                        </button>
                        <button
                            onClick={handleStop}
                            disabled={gameState !== 'running'}
                            className="flex items-center justify-center gap-2 py-4 rounded-xl bg-red-600 hover:bg-red-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-semibold shadow-lg shadow-red-900/20"
                        >
                            <Square size={20} fill="currentColor" /> Stop
                        </button>
                    </div>
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
                                style={{ imageRendering: 'pixelated' }}
                            />
                        ) : (
                            <div className="text-center space-y-4">
                                <div className="w-24 h-24 rounded-full bg-slate-900/80 flex items-center justify-center mx-auto border border-slate-700">
                                    <Ghost className="w-10 h-10 text-slate-500" />
                                </div>
                                <div>
                                    <p className="text-slate-400 text-lg">Ready to initialize environment</p>
                                    <p className="text-slate-600 text-sm">Select a game and algorithm to begin</p>
                                </div>
                            </div>
                        )}

                        {/* Overlay Stats */}
                        <div className="absolute top-4 right-4 flex gap-3">
                            <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/10 text-xs font-mono text-emerald-400">
                                FPS: 60
                            </div>
                            <div className="px-3 py-1.5 rounded-lg bg-black/60 backdrop-blur border border-white/10 text-xs font-mono text-blue-400">
                                Frames: {stats.length} {/* Approximate frame count via steps, or use new state */}
                            </div>
                        </div>
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 h-80">
                            <h3 className="text-slate-200 font-semibold mb-6 flex items-center gap-2">
                                <Activity size={18} className="text-blue-400" /> Training Rewards
                            </h3>
                            <ResponsiveContainer width="100%" height="80%">
                                <LineChart data={stats}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <XAxis dataKey="steps" hide />
                                    <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                                        itemStyle={{ color: '#fff' }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="reward"
                                        stroke="#8b5cf6"
                                        strokeWidth={3}
                                        dot={false}
                                        activeDot={{ r: 6, fill: '#8b5cf6' }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        <div className="p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50 h-80">
                            <h3 className="text-slate-200 font-semibold mb-4">Agent Status</h3>
                            <div className="space-y-4">
                                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                                    <span className="text-slate-400 text-sm">Status</span>
                                    <span className={cn("px-2 py-0.5 rounded text-xs font-bold uppercase",
                                        gameState === 'running' ? "bg-emerald-500/20 text-emerald-400" : "bg-slate-700/50 text-slate-400"
                                    )}>
                                        {gameState}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                                    <span className="text-slate-400 text-sm">Total Steps</span>
                                    <span className="font-mono text-blue-400">
                                        {stats.length > 0 ? stats[stats.length - 1].steps : 0}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/50 border border-slate-800">
                                    <span className="text-slate-400 text-sm">Last Reward</span>
                                    <span className="font-mono text-purple-400">
                                        {stats.length > 0 ? stats[stats.length - 1].reward.toFixed(2) : 0}
                                    </span>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;
