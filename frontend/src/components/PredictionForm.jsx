import React, { useState } from 'react';
import { TEAMS, CITIES } from '../constants';
import { predictMatch } from '../services/api';

const PredictionForm = ({ onResult }) => {
    const [formData, setFormData] = useState({
        batting_team: TEAMS[0],
        bowling_team: TEAMS[1],
        city: CITIES[0],
        target: '',
        score: '',
        wickets: '',
        overs: ''
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            const data = {
                ...formData,
                target: Number(formData.target),
                score: Number(formData.score),
                wickets: Number(formData.wickets),
                overs: Number(formData.overs)
            };
            const result = await predictMatch(data);
            onResult(result);
        } catch (err) {
            setError('Failed to get prediction. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card w-full max-w-xl bg-white/10 backdrop-blur-md shadow-2xl border border-white/20">
            <div className="card-body p-8">
                <div className="text-center mb-6">
                    <h2 className="text-3xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-blue-500">Match Details</h2>
                    <p className="text-base-content/70 mt-1">Enter current match scenario</p>
                </div>

                {error && <div className="alert alert-error shadow-lg mb-4 text-white"><span>{error}</span></div>}

                <form onSubmit={handleSubmit} className="space-y-6">

                    {/* Teams Selection */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                        <div className="form-control w-full group">
                            <label className="label"><span className="label-text font-bold text-base group-hover:text-primary transition-colors">Batting Team</span></label>
                            <select name="batting_team" className="select select-bordered select-primary w-full bg-base-100 focus:ring-2 focus:ring-primary focus:outline-none" value={formData.batting_team} onChange={handleChange}>
                                {TEAMS.map(team => <option key={team} value={team}>{team}</option>)}
                            </select>
                        </div>

                        <div className="form-control w-full group">
                            <label className="label"><span className="label-text font-bold text-base group-hover:text-secondary transition-colors">Bowling Team</span></label>
                            <select name="bowling_team" className="select select-bordered select-secondary w-full bg-base-100 focus:ring-2 focus:ring-secondary focus:outline-none" value={formData.bowling_team} onChange={handleChange}>
                                {TEAMS.map(team => <option key={team} value={team}>{team}</option>)}
                            </select>
                        </div>
                    </div>

                    {/* City Selection */}
                    <div className="form-control w-full group">
                        <label className="label"><span className="label-text font-bold text-base group-hover:text-accent transition-colors">Venue</span></label>
                        <select name="city" className="select select-bordered select-accent w-full bg-base-100 focus:ring-2 focus:ring-accent focus:outline-none" value={formData.city} onChange={handleChange}>
                            {CITIES.map(city => <option key={city} value={city}>{city}</option>)}
                        </select>
                    </div>

                    {/* Numeric Inputs */}
                    <div className="grid grid-cols-2 gap-5">
                        <div className="form-control group">
                            <label className="label"><span className="label-text font-semibold group-hover:text-info transition-colors">Target</span></label>
                            <input type="number" name="target" placeholder="180" className="input input-bordered input-info w-full bg-base-100 focus:ring-2 focus:ring-info focus:outline-none placeholder-gray-400 no-spinner" value={formData.target} onChange={handleChange} required min="0" />
                        </div>
                        <div className="form-control group">
                            <label className="label"><span className="label-text font-semibold group-hover:text-success transition-colors">Score</span></label>
                            <input type="number" name="score" placeholder="100" className="input input-bordered input-success w-full bg-base-100 focus:ring-2 focus:ring-success focus:outline-none placeholder-gray-400 no-spinner" value={formData.score} onChange={handleChange} required min="0" />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-5">
                        <div className="form-control group">
                            <label className="label"><span className="label-text font-semibold group-hover:text-warning transition-colors">Wickets</span></label>
                            <input type="number" name="wickets" placeholder="3" className="input input-bordered input-warning w-full bg-base-100 focus:ring-2 focus:ring-warning focus:outline-none placeholder-gray-400 no-spinner" value={formData.wickets} onChange={handleChange} required min="0" max="10" />
                        </div>
                        <div className="form-control group">
                            <label className="label"><span className="label-text font-semibold group-hover:text-error transition-colors">Overs</span></label>
                            <input type="number" name="overs" placeholder="10.4" className="input input-bordered input-error w-full bg-base-100 focus:ring-2 focus:ring-error focus:outline-none placeholder-gray-400 no-spinner" value={formData.overs} onChange={handleChange} required min="0" max="20" step="0.1" />
                        </div>
                    </div>

                    <div className="form-control mt-8">
                        <button type="submit" className={`btn btn-primary btn-block text-lg shadow-xl hover:scale-[1.02] transform transition-all ${loading ? 'loading' : ''}`} disabled={loading}>
                            {loading ? 'Crunching Numbers...' : 'Predict Outcome'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default PredictionForm;
