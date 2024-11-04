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
 *    React.JSX.Element InputArea()
 * 
 * DESCRIPTION
 *    This function exports the text box that the user
 *    will enter the text they want to be analyzed in.
 */
export default function InputArea() {
    return (
        <textarea rows="5">Enter your text here</textarea>
    );
}