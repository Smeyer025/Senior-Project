import "./AnalysisContainer.css";
import InputArea from "./InputArea/InputArea";
import DropDown from "./DropDown/DropDown";
import SubmitButton from "./SubmitButton/SubmitButton";

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