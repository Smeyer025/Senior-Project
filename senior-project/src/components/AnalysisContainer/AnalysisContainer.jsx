/**
 * AnalysisContainer Component: Container for input elements
 */

import "./AnalysisContainer.css";
import InputArea from "./InputArea/InputArea";
import DropDown from "./DropDown/DropDown";
import SubmitButton from "./SubmitButton/SubmitButton";

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
export default function AnalysisContainer(){
    return (
        <div id="AnalysisContainer">
            <InputArea />
            <DropDown type="Model"></DropDown>
            <DropDown type="Dataset"></DropDown>
            <SubmitButton />
        </div>
    );
}