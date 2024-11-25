/**
 * OutputContainer Component: Container for output
 */

import "./OutputContainer.css";

/**
 * OutputContainer()
 * 
 * NAME
 *    OutputContainer() - Handles layout for output from analysis
 * 
 * SYNOPSIS
 *    React.JSX.Element DropDown(props)
 *      props --> properties
 *          props.children --> Text to be shown in OutputContainer
 * 
 * DESCRIPTION
 */
export default function OutputContainer(props) {
    const determineColor = (children) => {
        console.log(`children: ${children}`);
        if (children == "positive") {
            return "green";
        } else if (children == "negative") {
            return 'red';
        } else {
            return 'black';
        }
    };

    return (
        <div className="OC">
            <h1 style={{color: determineColor(props.children)}}>{props.children}</h1>
        </div>
    );
};