const express = require("express");
const dotenv = require("dotenv");
const fs = require('fs');
const url = require('url');
const { exec } = require('child_process');


dotenv.config();
const PORT = process.env.PORT;
const app = express();


app.set("view engine", "ejs");

app.use(express.static("public"));
app.use(express.urlencoded({ extended: true }));


let globalData;

app.get("/", (req, res) => {
    res.render("index");
});

app.get("/output", (req, res) => {
    console.log("[LOG: FINALE] ");
    console.log(globalData);
    res.render("output", {
        globalData
    });
});

app.post("/", (req, res) => {
    // console.log(req.body.text_input);
    // let textData = "Well, that's just fantastic! I love getting stuck in traffic. Brilliant idea, let's have a meeting to decide when to schedule the next meeting. Sure, I'd love to work late on a Friday night. What could be more fun? Oh, great! Another flat tire on my way to work.";

    let textData = req.body.text_input;

    let sentences = textData.split(/[.!?]/).filter(sentence => sentence.trim() !== '');

    let jsonObject = { sentences };

    let jsonString = JSON.stringify(jsonObject, null, 2);
    fs.writeFileSync('output.json', jsonString);

    console.log('JSON file has been saved.');

    const command = 'python main.py';

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing command: ${error}`);
            return;
        }

        console.log(`Output: ${stdout}`);

        const jsonString = fs.readFileSync('predicted.json', 'utf8');

        console.log('Predicted JSON file has been saved.');

        // Parse the JSON string into a JavaScript object
        const jsonObject = JSON.parse(jsonString);

        // console.log(jsonObject);

        globalData = jsonObject;

        res.redirect("output");
        console.error(`Errors: ${stderr}`); 
    });

    exec("del predicted.json", (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing command: ${error}`);
            return;
        }

        console.log(`Output: ${stdout}`);
        console.error(`Errors: ${stderr}`); 
    });
});

//404 Error Block
app.use((req, res) => {
    res.status(404).render("404");
});

app.listen(PORT, () => {
    console.log(`[LOG] Connected and Listening on [PORT]: ${PORT}`);
});