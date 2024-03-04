const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
let bestStocks = null; // Variable to store the list of best stocks

// Enable CORS for all routes
app.use(cors());

// Serve static files from the 'static' directory
app.use(express.static('static'));

// Define a route for the root URL
app.get("/", (req, res) => {
    res.send("Welcome to PriceProphet");
});

// Define a route to start analysis
app.get("/start_analysis", (req, res) => {
    const pythonProcess = spawn('python', ['test_script.py']);

    pythonProcess.stdout.on('data', (data) => {
        console.log(`Python script output: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python script errors: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            console.error(`Python script process exited with code ${code}`);
            res.status(500).json({ error: 'Failed to start analysis' });
        } else {
            console.log('Analysis started successfully');
            res.json({ message: 'Analysis started successfully' });
        }
    });
});

// Define a route to fetch the best stocks
app.get("/best_stocks", (req, res) => {
    // Check if best stocks data is available
    if (bestStocks) {
        // Send the best stocks data to the client
        res.json({ bestStocks });
    } else {
        // If best stocks data is not available, send a message to the client
        res.status(404).json({ error: 'Best stocks data not available' });
    }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});