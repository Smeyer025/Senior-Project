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
 * 
 * DESCRIPTION
 *    This function exports the button that triggers the 
 *    analysis based on chosen parameters and inputted text
 */
export default function SubmitButton(props) {
    return (
        <button onClick={props.onClick} className="inline">Submit</button>
    );
}