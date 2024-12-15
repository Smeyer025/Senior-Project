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
 *    React.JSX.Element FileInput(a_props)
 *      a_props --> properties
 *          a_props.onChange --> The function that triggers when a new file
 *                             is uploaded
 * 
 * DESCRIPTION
 *    This function exports an input component that takes in datasets that 
 *    the user wants to train a sentiment analysis model on
 * 
 * RETURNS
 *    Returns the file input jsx element
 */
export default function FileInput(a_props) {
    return (
        <input type="file" onChange={a_props.onChange} className="inline"></input>
    );
}