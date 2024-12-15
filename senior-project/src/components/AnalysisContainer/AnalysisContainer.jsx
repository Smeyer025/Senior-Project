/**
 * AnalysisContainer Component: Container for input elements
 */

import { useState } from 'react';
import "./AnalysisContainer.css";
import InputArea from "./InputArea/InputArea";
import DropDown from "./DropDown/DropDown";
import FileInput from './FileInput/FileInput';
import SubmitButton from "./SubmitButton/SubmitButton";
import OutputContainer from '../OutputContainer/OutputContainer';

var Models = [
    "Logistic Regression",
    "Support Vector Machine",
    "Random Forest",
    "K Nearest Neighbors",
    "Voting Classifier"
];

var Datasets = [
    "Airline Reviews",
    "Drug Reviews",
    "Hotel Reviews",
    "Movie Reviews",
    "Social Media"
];

/**
 * AnalysisContainer()
 * 
 * NAME
 *    AnalysisContainer() - Handles layout for input elements
 * 
 * SYNOPSIS
 *    React.JSX.Element AnalysisContainer()
 * 
 * DESCRIPTION
 *    This function exports the layout for the elements
 *    that contribute to text analysis: the text input area,
 *    the model dropdown menu, the dataset dropdown menu, the
 *    input element that takes in an uploaded file, the button
 *    that triggers the import and makes the new dataset available
 *    and the submit button
 * 
 * RETURNS
 *    Returns the analysis container jsx element
 */
export default function AnalysisContainer() {    
    const [text, setText] = useState("Enter your text here");
    const [selectedValue1, setSelectedValue1] = useState('Logistic Regression');
    const [selectedValue2, setSelectedValue2] = useState('Airline Reviews');
    const [file, setFile] = useState();
    const [output, setOutput] = useState('Sentiment Output');

    const handleChange1 = (a_event) => {
        const value = a_event.target.value;
        setSelectedValue1(value);
        setOutput("Sentiment Output");
    };

    const handleChange2 = (a_event) => {
        console.log(Object.hasOwn(a_event, "target"));
        if(Object.hasOwn(a_event, "target")) {
            const value = a_event.target.value;
            setSelectedValue2(value);
        }
        else {
            setSelectedValue2(a_event);
        }
        setOutput("Sentiment Output");
    };

    const handleInput = (a_event) => {
        const value = a_event.target.value;
        setText(value);
        setOutput("Sentiment Output");
    };

    const handleClick = async () => {
        setOutput("Loading...");
        try {
            const response = await fetch(`http://localhost:5000/predict?datasetChoice=${selectedValue2.replaceAll(" ", "")}&modelType=${selectedValue1.replaceAll(" ", "")}&text="${text}"`)
            const data = await response.json();
            console.log(`http://localhost:5000/predict?datasetChoice=${selectedValue2.replaceAll(" ", "")}&modelType=${selectedValue1.replaceAll(" ", "")}&text="${text}"`)

            handleOutput(data);
        } catch (error) {
            console.error('Error in data fetch: ', error);
            handleOutput("Error, check console");
        }
    };

    const handleOutput = async (prediction) => {
        setOutput(prediction);
    }

    const handleFile = (a_event) => {
        var file = a_event.target.files[0];
        setFile(file);
        console.log(file.name);
        handleOutput("Sentiment Output");
    }

    const handleSubmitFile = async (a_event) => {
        a_event.preventDefault();

        const form = new FormData();
        form.append('file', file);
        form.append('fileName', file.name);

        var splitAnswers = [];
        while (splitAnswers.length != 2) {
            var promptAnswer = window.prompt("Enter the text column and the sentiment column, separated by a comma. (ex: review,sentiment)");
            var splitAnswers = promptAnswer.split(',');
        }

        try {
            const response = await fetch(`http://localhost:5000/uploadFile?text="${splitAnswers[0].replaceAll(" ", "")}"&sentiment="${splitAnswers[1].replaceAll(" ", "")}"`, {
                method: 'POST',
                body: form
            })
            const data = await response.json();
            if(data == "Invalid filetype, upload .csv") {
                handleOutput(data);
            }
            else {
                Datasets.push(data);
                handleChange2(data);
            }
        } catch (error) {
            console.error('Error in upload: ', error);
            handleOutput("Error, check console");
        }
    }

    return (
        <>
            <div id="AnalysisContainer">
                <InputArea value={text} onChange={handleInput}></InputArea>
                <DropDown value={selectedValue1} type="Model" list={Models} onChange={handleChange1}></DropDown>
                <DropDown value={selectedValue2} type="Dataset" list={Datasets} onChange={handleChange2}></DropDown>
                <FileInput onChange={handleFile}></FileInput>
                <SubmitButton onClick={handleSubmitFile}>Import Dataset</SubmitButton>
                <SubmitButton onClick={handleClick}>Submit</SubmitButton>
            </div>
            <OutputContainer>{output}</OutputContainer>
        </>
    );
}