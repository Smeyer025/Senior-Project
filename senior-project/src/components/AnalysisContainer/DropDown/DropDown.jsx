/**
 * DropDown Component: Modular drop down menu component
 */

import "./DropDown.css";

/**
 * DropDown()
 * 
 * NAME
 *    DropDown() - Handles layout for dropdown elements
 * 
 * SYNOPSIS
 *    React.JSX.Element DropDown(props)
 *      props --> properties
 *          props.type     --> The type of menu to output in dropdown menu
 *          props.value    --> The currently selected menu item
 *          props.onChange --> The function that triggers when a new value
 *                             is selected from the menu
 *          props.list     --> The array of data to output in the menu
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