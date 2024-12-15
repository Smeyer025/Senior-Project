/**
 * InputArea Component: Text box for input
 */

import "./InputArea.css";

/**
 * InputArea()
 * 
 * NAME
 *    InputArea() - Text box for user input
 * 
 * SYNOPSIS
 *    React.JSX.Element InputArea(a_props)
 *      a_props --> properties
 *          a_props.value --> Current text in box
 *          a_props.onChange --> The function that triggers when a new value
 *                             is selected from the menu
 * 
 * DESCRIPTION
 *    This function exports the text box that the user
 *    will enter the text they want to be analyzed in.
 * 
 * RETURNS
 *    Returns the input area jsx element
 */
export default function InputArea(a_props) {
    return (
        <textarea rows="5" value={a_props.value} onChange={a_props.onChange}></textarea>
    );
}