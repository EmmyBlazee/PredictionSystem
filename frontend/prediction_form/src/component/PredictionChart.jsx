import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  PieChart, Pie, Cell, Legend, ResponsiveContainer
} from 'recharts';
import { Paper, Typography, Button, Box } from '@mui/material';

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7f50'];

const PredictionChart = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    const fetchData = () => {
      axios.get('http://localhost:8000/summary')
        .then(res => setData(res.data))
        .catch(err => console.error("Error fetching summary:", err));
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleReset = async () => {
    try {
      await axios.delete('http://localhost:8000/reset-logs');
      setData([]);
    } catch (err) {
      console.error("Error resetting logs:", err);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5">Prediction Statistics</Typography>
        <Button variant="outlined" color="error" onClick={handleReset}>Reset Logs</Button>
      </Box>

      <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: 'center' }}>
        <ResponsiveContainer width="45%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="disease" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#1976d2" />
          </BarChart>
        </ResponsiveContainer>

        <ResponsiveContainer width="45%" height={300}>
          <PieChart>
            <Pie data={data} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={90} label>
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
};

export default PredictionChart;