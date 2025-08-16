import React, { useState, useEffect } from 'react';
import {
  Container,
  Box,
  Typography,
  Button,
  Paper,
  Grid,
  TextField,
  FormControl,
  Select,
  MenuItem,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Backdrop,
} from '@mui/material';
import { createTheme, ThemeProvider, useTheme } from '@mui/material/styles';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Refresh } from '@mui/icons-material';

// Import Firebase modules
import { initializeApp } from 'firebase/app';
import { getAuth, signInAnonymously, signInWithCustomToken, onAuthStateChanged } from 'firebase/auth';
import { getFirestore, collection, onSnapshot, addDoc, serverTimestamp, query, getDocs, writeBatch } from 'firebase/firestore';

// Define the Material-UI theme for the entire application.
const theme = createTheme({
  palette: {
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#ff9800',
    },
    background: {
      default: '#f4f6f8',
      paper: '#ffffff',
    },
    success: {
      main: '#4caf50',
    },
    error: {
      main: '#f44336',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.05)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 50,
          textTransform: 'none',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0px 2px 10px rgba(0, 0, 0, 0.1)',
          },
        },
      },
    },
  },
});

/**
 * A React component to visualize SHAP explanation data using Material-UI.
 * It displays features as a bar chart, showing their impact on the prediction.
 *
 * @param {Object} props
 * @param {Array<Object>} props.positiveFeatures - Features pushing the prediction higher.
 * @param {Array<Object>} props.negativeFeatures - Features pushing the prediction lower.
 */
const SHAPExplanationVisualizer = ({ positiveFeatures = [], negativeFeatures = [] }) => {
  const theme = useTheme();

  // Combine all features to find the maximum absolute SHAP value for scaling the bars.
  const allFeatures = [...positiveFeatures, ...negativeFeatures];
  const maxShapValue = allFeatures.length > 0
    ? Math.max(...allFeatures.map(f => Math.abs(f.shap_value)))
    : 1;

  const renderFeatureBar = (feature, isPositive) => {
    const barColor = isPositive ? theme.palette.success.main : theme.palette.error.main;
    const barWidth = (Math.abs(feature.shap_value) / maxShapValue) * 100;
    const sign = isPositive ? '+' : '';

    return (
      <Box key={feature.feature_name} sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Typography variant="body2" sx={{ width: '40%', textAlign: 'right', pr: 2, textTransform: 'capitalize' }}>
          {feature.feature_name.replace(/_/g, ' ')}
        </Typography>
        <Box sx={{ width: '60%', display: 'flex', alignItems: 'center' }}>
          <Box
            sx={{
              height: 20,
              backgroundColor: barColor,
              borderRadius: 1,
              transition: 'width 0.3s ease-in-out',
            }}
            style={{ width: `${barWidth}%` }}
          ></Box>
          <Typography variant="caption" sx={{ ml: 1, fontWeight: 'bold' }}>
            {sign}{feature.shap_value.toFixed(2)}
          </Typography>
        </Box>
      </Box>
    );
  };

  return (
    <Paper elevation={4} sx={{ p: 4, width: '100%', maxWidth: 1200, mt: 4 }}>
      <Typography variant="h5" component="h3" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold' }}>
        SHAP Explanation
      </Typography>
      {allFeatures.length > 0 ? (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" color="success.main" gutterBottom>
              Features Pushing Towards Positive Prediction
            </Typography>
            {positiveFeatures.map(feature => renderFeatureBar(feature, true))}
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" color="error.main" gutterBottom>
              Features Pushing Towards Negative Prediction
            </Typography>
            {negativeFeatures.map(feature => renderFeatureBar(feature, false))}
          </Grid>
        </Grid>
      ) : (
        <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
          No explanation data available.
        </Typography>
      )}
    </Paper>
  );
};


/**
 * A component to display a pie chart of the prediction history.
 *
 * @param {Object} props
 * @param {Array<string>} props.predictionHistory - An array of disease names.
 */
const PredictionPieChart = ({ predictionHistory }) => {
  const theme = useTheme();

  // Count the occurrences of each disease in the history.
  const diseaseCounts = predictionHistory.reduce((acc, disease) => {
    acc[disease] = (acc[disease] || 0) + 1;
    return acc;
  }, {});

  // Format the data for recharts
  const pieChartData = Object.keys(diseaseCounts).map(disease => ({
    name: disease.charAt(0).toUpperCase() + disease.slice(1),
    value: diseaseCounts[disease],
  }));

  // Assign a color to each disease
  const COLORS = [theme.palette.primary.main, theme.palette.success.main, theme.palette.error.main, theme.palette.secondary.main];

  return (
    <Paper elevation={4} sx={{ p: 4, width: '100%', maxWidth: 600, mx: 'auto' }}>
      <Typography variant="h5" component="h3" gutterBottom sx={{ textAlign: 'center', fontWeight: 'bold' }}>
        Most Common Predictions
      </Typography>
      {pieChartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={400}>
          <PieChart>
            <Pie
              data={pieChartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={150}
              fill="#8884d8"
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {pieChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      ) : (
        <Typography variant="body1" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
          Make a few predictions to see the chart.
        </Typography>
      )}
    </Paper>
  );
};

/**
 * The main component for the prediction form. It handles the input fields,
 * form submission, and displays the prediction results.
 *
 * @param {Object} props
 * @param {string} props.selectedDisease - The currently selected disease.
 * @param {Object} props.prediction - The prediction result from the backend.
 * @param {boolean} props.isLoading - Whether the prediction request is in progress.
 * @param {function(Object): void} props.handlePrediction - Function to handle the prediction request.
 */
const PredictionForm = ({ selectedDisease, prediction, shapData, isLoading, handlePrediction }) => {
  // Define the form fields based on the selected disease
  const forms = {
    heart: [
      { name: 'Age', label: 'Age (in years)', type: 'number', placeholder: 'e.g., 54', min: 18, max: 100 },
      { name: 'Sex', label: 'Sex', type: 'select', options: [{ value: 0, label: 'Female' }, { value: 1, label: 'Male' }] },
      { name: 'RestingBP', label: 'Resting Blood Pressure (mm/Hg)', type: 'number', placeholder: 'e.g., 120', min: 0 },
      { name: 'Cholesterol', label: 'Cholesterol (mg/dl)', type: 'number', placeholder: 'e.g., 200', min: 0 },
      { name: 'FastingBS', label: 'Fasting Blood Sugar > 120 mg/dl', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'RestingECG_Normal', label: 'Normal Resting ECG', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'RestingECG_ST', label: 'ST-T Wave Abnormality', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'MaxHR', label: 'Maximum Heart Rate Achieved', type: 'number', placeholder: 'e.g., 150', min: 0 },
      { name: 'ExerciseAngina', label: 'Exercise-Induced Angina', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Oldpeak', label: 'ST Depression Induced by Exercise', type: 'number', placeholder: 'e.g., 2.3', min: 0 },
      { name: 'ST_Slope_Flat', label: 'Flat ST Slope', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'ST_Slope_Up', label: 'Up-sloping ST Slope', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'ChestPainType_ATA', label: 'Chest Pain Type: Atypical Angina', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'ChestPainType_NAP', label: 'Chest Pain Type: Non-Anginal Pain', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'ChestPainType_TA', label: 'Chest Pain Type: Typical Angina', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
    ],
    diabetes: [
      { name: 'Pregnancies', label: 'Number of Pregnancies', type: 'number', placeholder: 'e.g., 6', min: 0 },
      { name: 'Glucose', label: 'Glucose Level (mg/dL)', type: 'number', placeholder: 'e.g., 148', min: 0 },
      { name: 'BloodPressure', label: 'Blood Pressure (mm/Hg)', type: 'number', placeholder: 'e.g., 72', min: 0 },
      { name: 'SkinThickness', label: 'Skin Thickness (mm)', type: 'number', placeholder: 'e.g., 35', min: 0 },
      { name: 'Insulin', label: 'Insulin Level (mu/mL)', type: 'number', placeholder: 'e.g., 0', min: 0 },
      { name: 'BMI', label: 'BMI', type: 'number', placeholder: 'e.g., 33.6', min: 0 },
      { name: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree Function', type: 'number', placeholder: 'e.g., 0.627', min: 0, step: 0.001 },
      { name: 'Age', label: 'Age (in years)', type: 'number', placeholder: 'e.g., 50', min: 0 },
    ],
    kidney: [
      { name: 'age', label: 'Age (in years)', type: 'number', placeholder: 'e.g., 48', min: 0 },
      { name: 'bp', label: 'Blood Pressure (mm/Hg)', type: 'number', placeholder: 'e.g., 80', min: 0 },
      { name: 'sg', label: 'Specific Gravity', type: 'number', step: 0.005, placeholder: 'e.g., 1.020', min: 0 },
      { name: 'al', label: 'Albumin', type: 'number', placeholder: 'e.g., 1', min: 0 },
      { name: 'su', label: 'Sugar', type: 'number', placeholder: 'e.g., 0', min: 0 },
      { name: 'bgr', label: 'Blood Glucose Random (mgs/dL)', type: 'number', placeholder: 'e.g., 121', min: 0 },
      { name: 'bu', label: 'Blood Urea (mgs/dL)', type: 'number', placeholder: 'e.g., 36', min: 0 },
      { name: 'sc', label: 'Serum Creatinine (mgs/dL)', type: 'number', placeholder: 'e.g., 1.2', min: 0 },
      { name: 'sod', label: 'Sodium (mEq/L)', type: 'number', placeholder: 'e.g., 136', min: 0 },
      { name: 'pot', label: 'Potassium (mEq/L)', type: 'number', placeholder: 'e.g., 4.7', min: 0 },
      { name: 'hemo', label: 'Hemoglobin (gms)', type: 'number', placeholder: 'e.g., 15.4', min: 0 },
      { name: 'pcv', label: 'Packed Cell Volume', type: 'number', placeholder: 'e.g., 44', min: 0 },
      { name: 'wc', label: 'White Blood Cells (cells/cmm)', type: 'number', placeholder: 'e.g., 7800', min: 0 },
      { name: 'rc', label: 'Red Blood Cells (millions/cmm)', type: 'number', placeholder: 'e.g., 5.2', min: 0 },
      { name: 'rbc', label: 'Red Blood Cells normal', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'pc', label: 'Pus Cells normal', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'pcc', label: 'Pus Cell Clumps present', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'ba', label: 'Bacteria present', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'htn', label: 'Hypertension', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'dm', label: 'Diabetes Mellitus', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'cad', label: 'Coronary Artery Disease', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'appet', label: 'Poor Appetite', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'pe', label: 'Pedal Edema', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'ane', label: 'Anemia', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
    ],
    hypertension: [
      { name: 'Age', label: 'Age (in years)', type: 'number', placeholder: 'e.g., 69', min: 0 },
      { name: 'Salt_Intake', label: 'Salt Intake (g/day)', type: 'number', placeholder: 'e.g., 8.0', min: 0 },
      { name: 'Stress_Score', label: 'Stress Score (0-10)', type: 'number', placeholder: 'e.g., 9', min: 0, max: 10 },
      { name: 'Sleep_Duration', label: 'Sleep Duration (hours)', type: 'number', placeholder: 'e.g., 6.4', min: 0, max: 24 },
      { name: 'BMI', label: 'BMI', type: 'number', placeholder: 'e.g., 25.8', min: 0 },
      { name: 'BP_History_Normal', label: 'BP History: Normal', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'BP_History_Prehypertension', label: 'BP History: Prehypertension', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Medication_Beta_Blocker', label: 'Medication: Beta Blocker', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Medication_Diuretic', label: 'Medication: Diuretic', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Medication_Other', label: 'Medication: Other', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Exercise_Level_Low', label: 'Exercise Level: Low', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Exercise_Level_Moderate', label: 'Exercise Level: Moderate', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Smoking_Status_Smoker', label: 'Smoking Status: Smoker', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Family_History_Yes', label: 'Family History: Yes', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
    ],
  };

  const [formData, setFormData] = useState({});

  useEffect(() => {
    if (selectedDisease) {
      // Initialize form data with empty strings or default values
      const initialData = forms[selectedDisease].reduce((acc, field) => {
        acc[field.name] = '';
        return acc;
      }, {});
      setFormData(initialData);
    }
  }, [selectedDisease]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    // Always parse the value to a float for numerical inputs,
    // but handle empty strings correctly.
    const newValue = value === '' ? '' : parseFloat(value);
    setFormData({
      ...formData,
      [name]: newValue,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handlePrediction(formData);
  };

  const currentFormFields = forms[selectedDisease] || [];

  return (
    <Paper elevation={4} sx={{ p: 4, width: '100%', maxWidth: 1200 }}>
      <Typography variant="h4" component="h2" gutterBottom align="center" sx={{ textTransform: 'capitalize' }}>
        {selectedDisease} Prediction
      </Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ mt: 3 }}>
        <Grid container spacing={3}>
          {currentFormFields.map((field) => (
            <Grid item xs={12} sm={6} md={4} key={field.name}>
              <Box sx={{ mb: 1 }}>
                <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                  {field.label}
                </Typography>
              </Box>
              {field.type === 'select' ? (
                <FormControl fullWidth required>
                  <Select
                    name={field.name}
                    value={formData[field.name] === '' ? '' : formData[field.name]}
                    onChange={handleInputChange}
                    displayEmpty
                  >
                    <MenuItem value=""><em>Select...</em></MenuItem>
                    {field.options.map(option => (
                      <MenuItem key={option.value} value={option.value}>{option.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              ) : (
                <TextField
                  fullWidth
                  required
                  name={field.name}
                  type="number"
                  value={formData[field.name] === undefined ? '' : formData[field.name]}
                  onChange={handleInputChange}
                  placeholder={field.placeholder}
                  inputProps={{
                    step: field.step || 'any',
                    min: field.min,
                    max: field.max,
                  }}
                />
              )}
            </Grid>
          ))}
          <Grid item xs={12} sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={isLoading}
              sx={{
                width: '100%', maxWidth: 300, py: 1.5,
                background: 'linear-gradient(45deg, #2196f3 30%, #21cbf3 90%)'
              }}
            >
              {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Predict & Explain'}
            </Button>
          </Grid>
        </Grid>
      </Box>
      {prediction && (
        <Box sx={{ mt: 4, p: 3, bgcolor: prediction.error ? '#fdecea' : '#e3f2fd', borderRadius: 2 }}>
          <Typography variant="h5" component="h3" gutterBottom>
            Prediction Result:
          </Typography>
          {prediction.error ? (
            <Typography color="error" sx={{ fontWeight: 'bold' }}>
              {prediction.error}
            </Typography>
          ) : (
            <>
              <Typography variant="h6">
                Result: <span style={{ fontWeight: 'bold' }}>{prediction.prediction === 1 ? 'Positive' : 'Negative'}</span>
              </Typography>
              <Typography variant="h6">
                Probability of having the disease: <span style={{ fontWeight: 'bold' }}>{(prediction.probability_has_disease * 100).toFixed(2)}%</span>
              </Typography>
            </>
          )}
        </Box>
      )}
    </Paper>
  );
};

// The main App component, which orchestrates the entire application.
const App = () => {
  // State for the currently selected disease and its associated data
  const [selectedDisease, setSelectedDisease] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [shapData, setShapData] = useState(null);

  // Loading states for prediction and SHAP explanation API calls
  const [isLoading, setIsLoading] = useState(false);
  const [isClearingHistory, setIsClearingHistory] = useState(false);
  const [clearHistoryMessage, setClearHistoryMessage] = useState({ open: false, title: '', content: '' });

  // Firebase states
  const [db, setDb] = useState(null);
  const [userId, setUserId] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [isAuthReady, setIsAuthReady] = useState(false);

  // A list of the available diseases to predict
  const availableDiseases = ['heart', 'diabetes', 'kidney', 'hypertension'];

  // 1. Firebase Initialization and Authentication
  useEffect(() => {
    // Check if Firebase config is available globally
    if (typeof __firebase_config !== 'undefined') {
      const firebaseConfig = JSON.parse(__firebase_config);
      const app = initializeApp(firebaseConfig);
      const auth = getAuth(app);
      const firestoreDb = getFirestore(app);
      setDb(firestoreDb);

      const unsubscribe = onAuthStateChanged(auth, async (user) => {
        if (user) {
          setUserId(user.uid);
        } else {
          try {
            if (typeof __initial_auth_token !== 'undefined' && __initial_auth_token) {
              await signInWithCustomToken(auth, __initial_auth_token);
            } else {
              await signInAnonymously(auth);
            }
            setUserId(auth.currentUser.uid);
          } catch (error) {
            console.error('Authentication failed:', error);
          }
        }
        setIsAuthReady(true);
      });

      return () => unsubscribe(); // Cleanup function for the listener
    } else {
      console.error('Firebase config is not available.');
      setIsAuthReady(true); // Ensure the app loads even without Firebase
    }
  }, []);

  // 2. Real-time Firestore Data Fetching
  useEffect(() => {
    if (!isAuthReady || !db || !userId) return;

    // The collection path for private data
    const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
    const collectionPath = `/artifacts/${appId}/users/${userId}/prediction_history`;
    const q = query(collection(db, collectionPath));

    const unsubscribe = onSnapshot(q, (querySnapshot) => {
      const history = [];
      querySnapshot.forEach((doc) => {
        const data = doc.data();
        if (data.prediction_result === 1) { // Only add positive predictions
          history.push(data.disease_name);
        }
      });
      setPredictionHistory(history);
    }, (error) => {
      console.error('Error fetching prediction history:', error);
    });

    return () => unsubscribe();
  }, [db, userId, isAuthReady]);

  /**
   * Handles the selection of a new disease.
   * Resets the prediction and explanation data.
   * @param {string} disease The name of the selected disease.
   */
  const handleDiseaseSelect = (disease) => {
    setSelectedDisease(disease);
    setPrediction(null);
    setShapData(null);
  };

  /**
   * Handles the form submission by making API calls to the backend.
   * This function is passed as a prop to the PredictionForm component.
   * It fetches both the prediction and the SHAP explanation in parallel.
   * @param {Object} formData The data from the prediction form.
   */
  const handlePrediction = async (formData) => {
    setIsLoading(true);
    setPrediction(null);
    setShapData(null);

    const backendUrl = 'https://predictionsystem-fwa7.onrender.com';

    try {
      // Create a promise for the prediction API call
      const predictionPromise = fetch(`${backendUrl}/predict/${selectedDisease}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      }).then(response => {
        if (!response.ok) {
          return response.json().then(errorData => {
            throw new Error(errorData.detail || 'Something went wrong with the prediction.');
          });
        }
        return response.json();
      });

      // Create a promise for the SHAP explanation API call
      const shapPromise = fetch(`${backendUrl}/explain/${selectedDisease}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      }).then(response => {
        if (!response.ok) {
          return response.json().then(errorData => {
            throw new Error(errorData.detail || 'Something went wrong with the SHAP explanation.');
          });
        }
        return response.json();
      });

      // Wait for both promises to resolve
      const [predictionResult, shapResult] = await Promise.allSettled([predictionPromise, shapPromise]);

      // Handle the prediction result
      if (predictionResult.status === 'fulfilled') {
        const result = predictionResult.value;
        setPrediction(result);

        // Save the prediction to Firestore
        if (db && userId) {
          const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
          const historyCollectionRef = collection(db, `/artifacts/${appId}/users/${userId}/prediction_history`);

          await addDoc(historyCollectionRef, {
            disease_name: selectedDisease,
            prediction_result: result.prediction,
            probability: result.probability_has_disease,
            formData,
            timestamp: serverTimestamp(),
          });
        }
      } else {
        setPrediction({ error: predictionResult.reason.message });
        console.error('Prediction Error:', predictionResult.reason);
      }

      // Handle the SHAP explanation result
      if (shapResult.status === 'fulfilled') {
        setShapData(shapResult.value);
      } else {
        setShapData({ error: shapResult.reason.message });
        console.error('SHAP Explanation Error:', shapResult.reason);
      }
    } catch (error) {
      console.error('API call failed:', error);
      setPrediction({ error: 'Failed to connect to the backend server.' });
      setShapData(null);
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Deletes all predictions for the current user from the Firestore database.
   * This function uses a batch write for efficiency.
   */
  const handleClearHistory = async () => {
    if (!db || !userId) return;

    setIsClearingHistory(true);
    const appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';
    const historyCollectionRef = collection(db, `/artifacts/${appId}/users/${userId}/prediction_history`);

    try {
      const q = query(historyCollectionRef);
      const querySnapshot = await getDocs(q);
      const batch = writeBatch(db);

      querySnapshot.forEach((doc) => {
        batch.delete(doc.ref);
      });

      await batch.commit();
      setClearHistoryMessage({ open: true, title: 'Success', content: 'Prediction history cleared successfully.' });
      console.log('Prediction history cleared successfully.');
    } catch (error) {
      setClearHistoryMessage({ open: true, title: 'Error', content: 'Failed to clear history. Please try again.' });
      console.error('Error clearing prediction history:', error);
    } finally {
      setIsClearingHistory(false);
    }
  };

  const handleCloseClearHistoryMessage = () => {
    setClearHistoryMessage(prev => ({ ...prev, open: false }));
  };

  if (!isAuthReady) {
    return (
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={true}
      >
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress color="inherit" />
          <Typography sx={{ mt: 2 }}>Connecting...</Typography>
        </Box>
      </Backdrop>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="lg" sx={{ my: 4 }}>
        <Box sx={{ textAlign: 'center', mb: 6 }}>
          <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 'bold', color: 'primary.main' }}>
            Medical Diagnosis Predictor
          </Typography>
          <Typography variant="h6" color="text.secondary">
            User ID: {userId}
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Select a disease below, enter the required medical data, and get an instant prediction.
          </Typography>
        </Box>

        {/* Disease selection buttons */}
        <Grid container spacing={3} justifyContent="center" sx={{ mb: 6 }}>
          {availableDiseases.map((disease) => (
            <Grid item key={disease} xs={12} sm={6} md={3}>
              <Button
                variant={selectedDisease === disease ? 'contained' : 'outlined'}
                fullWidth
                size="large"
                onClick={() => handleDiseaseSelect(disease)}
                sx={{ py: 2, textTransform: 'capitalize' }}
              >
                {disease}
              </Button>
            </Grid>
          ))}
        </Grid>

        {/* Dynamic content based on selected disease */}
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          {selectedDisease ? (
            <>
              {/* The PredictionForm component now receives all its necessary props */}
              <PredictionForm
                selectedDisease={selectedDisease}
                prediction={prediction}
                shapData={shapData}
                isLoading={isLoading}
                handlePrediction={handlePrediction}
              />
              {/* Render the SHAP and Pie charts only when data is available */}
              {shapData && !shapData.error && (
                <SHAPExplanationVisualizer
                  positiveFeatures={shapData.positive_features}
                  negativeFeatures={shapData.negative_features}
                />
              )}
              <Box sx={{ mt: 4, width: '100%' }}>
                <PredictionPieChart predictionHistory={predictionHistory} />
                <Box sx={{ textAlign: 'center', mt: 2 }}>
                  <Button variant="outlined" onClick={handleClearHistory} startIcon={<Refresh />} disabled={isClearingHistory}>
                    {isClearingHistory ? <CircularProgress size={24} /> : 'Clear History'}
                  </Button>
                </Box>
              </Box>
              {/* Modal for clearing history status */}
              <Dialog
                open={clearHistoryMessage.open}
                onClose={handleCloseClearHistoryMessage}
                aria-labelledby="clear-history-title"
              >
                <DialogTitle id="clear-history-title">{clearHistoryMessage.title}</DialogTitle>
                <DialogContent>
                  <Typography>{clearHistoryMessage.content}</Typography>
                </DialogContent>
                <DialogActions>
                  <Button onClick={handleCloseClearHistoryMessage} autoFocus>
                    OK
                  </Button>
                </DialogActions>
              </Dialog>
            </>
          ) : (
            <Paper elevation={4} sx={{ p: 4, textAlign: 'center', width: '100%', maxWidth: 600 }}>
              <Typography variant="h5" component="h2" gutterBottom>
                Welcome!
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Please select one of the disease prediction models above to get started.
              </Typography>
            </Paper>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
};

export default App;
