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
            {props.list.map((item) => <option>{item}</option>)}
        </select>
    );
}