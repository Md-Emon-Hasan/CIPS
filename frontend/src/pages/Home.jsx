import React, { useState } from 'react';
import PredictionForm from '../components/PredictionForm';
import Footer from '../components/Footer';

const Home = () => {
    const [result, setResult] = useState(null);

    return (
        <div className="flex flex-col min-h-screen">
            {/* Navbar */}
            <div className="navbar bg-base-100 shadow-md px-4 sm:px-8">
                <div className="flex-1">
                    <a className="btn btn-ghost normal-case text-xl flex items-center gap-2">
                        <span className="font-bold text-2xl">CIPS</span>
                    </a>
                </div>
                <div className="flex-none">
                    <ul className="menu menu-horizontal px-1">
                        <li><a href="https://github.com/Md-Emon-Hasan/CIPS" target="_blank">Source Code</a></li>
                    </ul>
                </div>
            </div>

            {/* Main Content */}
            <main className="flex-grow bg-base-200 py-10 px-4">
                <div className="max-w-4xl mx-auto flex flex-col items-center gap-8 animate-fade-in-down">

                    <div className="text-center space-y-2">
                        <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary mb-2 drop-shadow-sm">
                            IPL Win Predictor
                        </h1>
                        <p className="text-lg opacity-80 max-w-2xl mx-auto">
                            Advanced Machine Learning model to predict match outcomes based on real-time game situations.
                            Select teams, venue, and current score to get instant probability.
                        </p>
                    </div>

                    <div className="flex flex-col md:flex-row gap-8 w-full justify-center items-start mt-8">
                        <div className="w-full md:w-1/2 flex justify-center">
                            <PredictionForm onResult={setResult} />
                        </div>

                        {result && (
                            <div className="w-full md:w-1/2 card bg-base-100 shadow-xl animate-fade-in-up border border-base-300">
                                <div className="card-body items-center text-center">
                                    <h2 className="card-title text-2xl mb-6 border-b pb-2 w-full justify-center">Prediction Analysis</h2>

                                    <div className="w-full space-y-6">
                                        {/* Batting Team Prob */}
                                        <div>
                                            <div className="flex justify-between mb-1">
                                                <span className="font-semibold">{result.batting_team}</span>
                                                <span className="font-bold text-primary">{result.batting_team_probability}%</span>
                                            </div>
                                            <progress className="progress progress-primary w-full h-4" value={result.batting_team_probability} max="100"></progress>
                                        </div>

                                        {/* Bowling Team Prob */}
                                        <div>
                                            <div className="flex justify-between mb-1">
                                                <span className="font-semibold">{result.bowling_team}</span>
                                                <span className="font-bold text-secondary">{result.bowling_team_probability}%</span>
                                            </div>
                                            <progress className="progress progress-secondary w-full h-4" value={result.bowling_team_probability} max="100"></progress>
                                        </div>
                                    </div>

                                    <div className="stats shadow mt-8 w-full">
                                        <div className="stat place-items-center">
                                            <div className="stat-title">Win Probability</div>
                                            <div className="stat-value text-2xl text-primary">{Math.max(result.batting_team_probability, result.bowling_team_probability)}%</div>
                                            <div className="stat-desc">For leading team</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {!result && (
                            <div className="w-full md:w-1/2 hidden md:flex flex-col justify-center items-center text-center opacity-40 p-10 border-2 border-dashed border-base-300 rounded-box h-full min-h-[400px]">
                                <span className="text-6xl mb-4 text-base-content/20">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="currentColor"><path d="M4 19h4v-7H4v7zm6 0h4v-11h-4v11zm6 0h4v-15h-4v15zm2 2H2V3h2v17h16v1z" /></svg>
                                </span>
                                <p className="text-xl font-medium">Result will appear here</p>
                                <p className="text-sm">Enter match details on the left to analyze.</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            {/* Footer */}
            <Footer />
        </div>
    );
};

export default Home;
