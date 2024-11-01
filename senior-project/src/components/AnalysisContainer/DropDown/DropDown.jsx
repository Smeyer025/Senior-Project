import "./DropDown.css";

export default function DropDown(props) {
    return (
        <select class="inline" name={props.type} >
            {props.type === "Model" ? Models.map((item) => <option>{item}</option>) : Datasets.map((item) => <option>{item}</option>)}
        </select>
    );
}

const Models = [
    "Logistic Regression",
    "Support Vector Machine",
    "Random Forest/Decision Trees",
    "Naïve Bayes"
];

const Datasets = [
    "Airline Reviews",
    "Drug Reviews",
    "Hotel Reviews",
    "Movie Reviews",
    "Social Media"
];