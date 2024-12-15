/**
 * SubmitButton Component: Button that triggers analysis
 */

/**
 * SubmitButton()
 * 
 * NAME
 *    SubmitButton() - Button that triggers analysis
 *                     based on chosen parameters and 
 *                     inputted text
 * 
 * SYNOPSIS
 *    React.JSX.Element SubmitButton()
 *      a_props --> properties
 *          a_props.onClick  --> The function that triggers when the 
 *                             button is clicked
 *          a_props.children --> The text to be displayed on the button
 * 
 * DESCRIPTION
 *    This function exports the button that triggers the 
 *    analysis based on chosen parameters and inputted text
 * 
 * RETURNS
 *    Returns the submit button jsx element
 */
export default function SubmitButton(a_props) {
    return (
        <button onClick={a_props.onClick} className="inline">{a_props.children}</button>
    );
}