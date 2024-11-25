/**
 * FileInput Component: File input component
 */

/**
 * FileInput()
 * 
 * NAME
 *    FileInput() - Handles layout for file input button
 * 
 * SYNOPSIS
 *    React.JSX.Element FileInput(props)
 *      props --> properties
 *          props.onChange --> The function that triggers when a new file
 *                             is uploaded
 * 
 * DESCRIPTION
 *    This function exports an input component that takes in datasets that 
 *    the user wants to train a sentiment analysis model on
 */
export default function FileInput(props) {
    return (
        <input type="file" onChange={props.onChange} className="inline"></input>
    );
}