/**
 * MenuContainer Component: Container for all the MenuElements
 */

import MenuElement from "./MenuElement/MenuElement";
import OutputContainer from "../OutputContainer/OutputContainer";
import { useState } from "react";
import "./MenuContainer.css";

/**
 * MenuContainer()
 * 
 * NAME
 *    MenuContainer() - Handles layout for performance metrics buttons
 * 
 * SYNOPSIS
 *    React.JSX.Element MenuContainer()
 * 
 * DESCRIPTION
 *    This function exports a grid of MenuElements that make up 
 *    the performance metrics menu
 * 
 * RETURNS 
 *    Returns menu container jsx element
 */
export default function MenuContainer() {
    const [output, setOutput] = useState('Menu Output');

    const handleClicks = async (elem) => {
        setOutput("Loading...");
        try {
            const response = await fetch(`http://localhost:5000/${elem}`)
            const data = await response.json();
            setOutput(data);
        } catch (error) {
            console.error('Error in data fetch: ', error);
            setOutput("Error, check console");
        }
    }

    return (
        <>
            <div className="MenuBackground">
                <MenuElement className="Title"><h3>Performance Metrics</h3></MenuElement>
                <div className="MenuContainer">
                    <MenuElement className="MenuElement" onClick={() => handleClicks("accuracy")}>Accuracy</MenuElement>
                    <MenuElement className="MenuElement" onClick={() => handleClicks("precision")}>Precision</MenuElement>
                    <MenuElement className="MenuElement" onClick={() => handleClicks("recall")}>Recall</MenuElement>
                    <MenuElement className="MenuElement" onClick={() => handleClicks("f1score")}>F1 Score</MenuElement>
                    <MenuElement className="MenuElement" onClick={() => handleClicks("hamming_loss")}>Hamming loss</MenuElement>
                    <MenuElement className="MenuElement" onClick={() => handleClicks("kfold")}>Run 5-Fold Cross Validation</MenuElement>
                    <MenuElement className="MenuElement" onClick={() => handleClicks("confusion_matrix")}>Output Confusion Matrix</MenuElement>
                </div>
            </div>
            <OutputContainer>{output}</OutputContainer>
        </>
    );
}