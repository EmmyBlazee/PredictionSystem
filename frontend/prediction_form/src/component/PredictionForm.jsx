import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Grid,
  TextField,
  FormControl,
  Select,
  MenuItem,
  CircularProgress
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * A React component to visualize SHAP explanation data using Material-UI.
 * It displays features as a bar chart, showing their impact on the prediction.
 *
 * @param {Object} props
 * @param {Array<Object>} props.positiveFeatures - Features pushing the prediction higher.
 * @param {Array<Object>} props.negativeFeatures - Features pushing the prediction lower.
 */
export const SHAPExplanationVisualizer = ({ positiveFeatures = [], negativeFeatures = [] }) => {
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
export const PredictionPieChart = ({ predictionHistory }) => {
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
 * @param {Object} props.shapData - The SHAP explanation data from the backend.
 * @param {boolean} props.isLoading - Whether the prediction request is in progress.
 * @param {function(Object): void} props.handlePrediction - Function to handle the prediction request.
 */
const PredictionForm = ({ selectedDisease, prediction, shapData, isLoading, handlePrediction }) => {
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
      { name: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree Function', type: 'number', placeholder: 'e.g., 0.627', min: 0 },
      { name: 'Age', label: 'Age (in years)', type: 'number', placeholder: 'e.g., 50', min: 0 },
    ],
    kidney: [
      { name: 'Age_yrs', label: 'Age (in years)', type: 'number', placeholder: 'e.g., 48', min: 0 },
      { name: 'Blood_Pressure_mm_Hg', label: 'Blood Pressure (mm/Hg)', type: 'number', placeholder: 'e.g., 80', min: 0 },
      { name: 'Specific_Gravity', label: 'Specific Gravity', type: 'number', step: 0.005, placeholder: 'e.g., 1.020', min: 0 },
      { name: 'Albumin', label: 'Albumin', type: 'number', placeholder: 'e.g., 1', min: 0 },
      { name: 'Sugar', label: 'Sugar', type: 'number', placeholder: 'e.g., 0', min: 0 },
      { name: 'Blood_Glucose_Random_mgs_dL', label: 'Blood Glucose Random (mgs/dL)', type: 'number', placeholder: 'e.g., 121', min: 0 },
      { name: 'Blood_Urea_mgs_dL', label: 'Blood Urea (mgs/dL)', type: 'number', placeholder: 'e.g., 36', min: 0 },
      { name: 'Serum_Creatinine_mgs_dL', label: 'Serum Creatinine (mgs/dL)', type: 'number', placeholder: 'e.g., 1.2', min: 0 },
      { name: 'Sodium_mEq_L', label: 'Sodium (mEq/L)', type: 'number', placeholder: 'e.g., 136', min: 0 },
      { name: 'Potassium_mEq_L', label: 'Potassium (mEq/L)', type: 'number', placeholder: 'e.g., 4.7', min: 0 },
      { name: 'Hemoglobin_gms', label: 'Hemoglobin (gms)', type: 'number', placeholder: 'e.g., 15.4', min: 0 },
      { name: 'Packed_Cell_Volume', label: 'Packed Cell Volume', type: 'number', placeholder: 'e.g., 44', min: 0 },
      { name: 'White_Blood_Cells_cells_cmm', label: 'White Blood Cells (cells/cmm)', type: 'number', placeholder: 'e.g., 7800', min: 0 },
      { name: 'Red_Blood_Cells_millions_cmm', label: 'Red Blood Cells (millions/cmm)', type: 'number', placeholder: 'e.g., 5.2', min: 0 },
      { name: 'Red_Blood_Cells_normal', label: 'Red Blood Cells Normal', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Pus_Cells_normal', label: 'Pus Cells Normal', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Pus_Cell_Clumps_present', label: 'Pus Cell Clumps Present', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Bacteria_present', label: 'Bacteria Present', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Hypertension_yes', label: 'Hypertension', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Diabetes_Mellitus_yes', label: 'Diabetes Mellitus', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Coronary_Artery_Disease_yes', label: 'Coronary Artery Disease', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Appetite_poor', label: 'Poor Appetite', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Pedal_Edema_yes', label: 'Pedal Edema', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
      { name: 'Anemia_yes', label: 'Anemia', type: 'select', options: [{ value: 0, label: 'No' }, { value: 1, label: 'Yes' }] },
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
      const initialData = forms[selectedDisease].reduce((acc, field) => {
        acc[field.name] = '';
        return acc;
      }, {});
      setFormData(initialData);
    }
  }, [selectedDisease]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value === '' ? '' : parseFloat(value),
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
                    value={formData[field.name]}
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
                  value={formData[field.name]}
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
        <Box sx={{ mt: 4, p: 3, bgcolor: '#e3f2fd', borderRadius: 2 }}>
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

export default PredictionForm;
