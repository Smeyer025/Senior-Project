/**
 * AnalysisContainer Component: Container for input elements
 */

import { useState, useEffect } from 'react';
import "./AnalysisContainer.css";
import InputArea from "./InputArea/InputArea";
import DropDown from "./DropDown/DropDown";
import SubmitButton from "./SubmitButton/SubmitButton";
import OutputContainer from '../OutputContainer/OutputContainer';

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
 *    the model dropdown menu, the dataset dropdown menu,
 *    and the submit button
 */
export default function AnalysisContainer() {
    const [text, setText] = useState("Enter your text here");
    const [selectedValue1, setSelectedValue1] = useState('Logistic Regression');
    const [selectedValue2, setSelectedValue2] = useState('Airline Reviews');
    const [output, setOutput] = useState('Output');

    const handleChange1 = (event) => {
        const value = event.target.value;
        setSelectedValue1(value);
        //setText(`Model selected: ${value}`);
        console.log(value);
    };

    const handleChange2 = (event) => {
        const value = event.target.value;
        setSelectedValue2(value);
        //setText(`Dataset selected: ${value}`);
        console.log(value);
    };

    const handleInput = (event) => {
        const value = event.target.value;
        setText(value);
    };

    const handleClick = async () => {
        setOutput("Loading...");
        try {
            const response = await fetch(`http://127.0.0.1:5000/predict?datasetChoice=${selectedValue2.replaceAll(" ", "")}&modelType=${selectedValue1.replaceAll(" ", "")}&text="${text}"`)
            const data = await response.json();
            console.log(`http://127.0.0.1:5000/predict?datasetChoice=${selectedValue2.replaceAll(" ", "")}&modelType=${selectedValue1.replaceAll(" ", "")}&text="${text}"`)

            handleOutput(data);
        } catch (error) {
            console.error('Error in data fetch: ', error);
            handleOutput("Error, check console");
        }
    };

    const handleOutput = (prediction) => {
        setOutput(prediction);
    }

    return (
        <>
            <div id="AnalysisContainer">
                <InputArea value={text} onChange={handleInput}></InputArea>
                <DropDown value={selectedValue1} type="Model" onChange={handleChange1}></DropDown>
                <DropDown value={selectedValue2} type="Dataset" onChange={handleChange2}></DropDown>
                <SubmitButton onClick={handleClick} />
            </div>
            <OutputContainer>{output}</OutputContainer>
        </>
    );
}