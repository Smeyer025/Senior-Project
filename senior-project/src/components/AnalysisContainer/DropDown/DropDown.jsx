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
 *    React.JSX.Element DropDown(a_props)
 *      a_props --> properties
 *          a_props.type     --> The type of menu to output in dropdown menu
 *          a_props.value    --> The currently selected menu item
 *          a_props.onChange --> The function that triggers when a new value
 *                             is selected from the menu
 *          a_props.list     --> The array of data to output in the menu
 * 
 * DESCRIPTION
 *    This function exports a modular dropdown menu, that either
 *    has the list of model types or the list of datasets based on
 *    what the user chooses
 * 
 * RETURNS
 *    Returns the drop down menu jsx element
 */
export default function DropDown(a_props) {
    return (
        <select value={a_props.value} name={a_props.type} onChange={a_props.onChange} className="inline">
            {a_props.list.map((item) => <option>{item}</option>)}
        </select>
    );
}