/**
 * DropDown Component: Modular drop down menu component
 */

import "./DropDown.css";
import { useState } from 'react';

/**
 * DropDown()
 * 
 * NAME
 *    DropDown() - Handles layout for input elements
 * 
 * SYNOPSIS
 *    React.JSX.Element DropDown(props)
 *      props --> properties
 *          props.type --> Contains type of menu to output in dropdown menu
 * 
 * DESCRIPTION
 *    This function exports a modular dropdown menu, that either
 *    has the list of model types or the list of datasets based on
 *    what the user chooses
 */
export default function DropDown(props) {
    return (
        <select value={props.value} name={props.type} onChange={props.onChange} className="inline">
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