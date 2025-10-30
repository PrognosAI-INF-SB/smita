// === Sensor Trend (Static Example from Last Input) ===
const sensorCtx = document.getElementById('sensorChart').getContext('2d');
const sensorChart = new Chart(sensorCtx, {
    type: 'line',
    data: {
        labels: Array.from({length: 21}, (_, i) => `S${i+1}`),
        datasets: [{
            label: 'Sensor Values',
            data: [], // Filled dynamically
            borderColor: '#0077b6',
            backgroundColor: 'rgba(0,119,182,0.1)',
            tension: 0.3
        }]
    },
    options: {
        responsive: true,
        scales: { y: { beginAtZero: true } }
    }
});

// === Health Score Over Time ===
const healthCtx = document.getElementById('healthChart').getContext('2d');
const healthChart = new Chart(healthCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Health Score',
            data: [],
            borderColor: '#00b4d8',
            backgroundColor: 'rgba(0,180,216,0.1)',
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        animation: false,
        scales: { y: { beginAtZero: true, max: 100 } }
    }
});

// === Auto-fetch live data from Flask ===
async function updateCharts() {
    const response = await fetch('/data');
    const jsonData = await response.json();

    // Update Health History Chart
    healthChart.data.labels = jsonData.map(row => row.Timestamp);
    healthChart.data.datasets[0].data = jsonData.map(row => row.Health_Score);
    healthChart.update();
}

// Refresh every 5 seconds
setInterval(updateCharts, 5000);
updateCharts();
